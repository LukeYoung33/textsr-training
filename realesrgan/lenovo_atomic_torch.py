"""Device-resident candidate port of frozen OpenCV atoms; certification is separate.

Images are RGB BCHW floats. Random fields can be supplied for paired numerical
tests; normal execution creates them on the input device using a Torch generator.
"""
import math
import torch
from torch.nn import functional as F

VERSION = 'lenovo_torch_c3_edge_v2'


class ReferenceJpeg(torch.nn.Module):
    """Torch DCT codec with integer JPEG quantizers and smooth 4:2:0 expansion."""
    def __init__(self):
        super().__init__()
        from basicsr.utils.diffjpeg import DiffJPEG
        self.codec = DiffJPEG(differentiable=False)
        self.register_buffer('base_y', self.codec.compress.y_quantize.y_table.detach().clone())
        self.register_buffer('base_c', self.codec.compress.c_quantize.c_table.detach().clone())
        class Upsample(torch.nn.Module):
            def forward(self, y, cb, cr):
                size=y.shape[-2:]
                cb=F.interpolate(cb[:,None],size=size,mode='bilinear',align_corners=False)[:,0]
                cr=F.interpolate(cr[:,None],size=size,mode='bilinear',align_corners=False)[:,0]
                return torch.stack([y,cb,cr],-1)
        self.codec.decompress.chroma=Upsample()

    def forward(self, x, quality):
        q=max(1,min(100,int(quality))); scale=5000/q if q<50 else 200-2*q
        for kind,base in [('y',self.base_y),('c',self.base_c)]:
            table=((base*scale+50)/100).floor().clamp(1,255)
            for block in [getattr(self.codec.compress,kind+'_quantize'),getattr(self.codec.decompress,kind+'_dequantize')]:
                setattr(block,kind+'_table',torch.nn.Parameter(table,requires_grad=False))
        h,w=x.shape[-2:]; padded=F.pad(x,(0,(-w)%16,0,(-h)%16),mode='replicate')
        y,cb,cr=self.codec.compress(padded,factor=1.)
        result=self.codec.decompress(y,cb,cr,*padded.shape[-2:],factor=1.)
        return result[:,:,:h,:w]


def quantize(x):
    return x.clamp(0, 1).mul(255).round().div(255)


def reflect_filter(x, kernel):
    r = kernel.shape[-1] // 2
    # Index folding supports kernels larger than tiny images as well.
    def indices(n):
        if n == 1:
            return torch.zeros(n + kernel.shape[-1]-1, device=x.device, dtype=torch.long)
        i = torch.arange(-r, n+kernel.shape[-1]-1-r, device=x.device).remainder(2*n-2)
        return torch.minimum(i, 2*n-2-i)
    padded = x.index_select(-2, indices(x.shape[-2])).index_select(-1, indices(x.shape[-1]))
    return F.conv2d(padded, kernel[None, None].expand(x.shape[1], 1, -1, -1), groups=x.shape[1])


def gaussian(x, sigma, byte_input=False):
    # OpenCV automatic support is approximately 3 sigma for uint8, 4 for float.
    size = int(round(sigma * (6 if byte_input else 8) + 1)) | 1
    a = torch.arange(size, device=x.device, dtype=x.dtype) - size//2
    k = torch.exp(-a*a/(2*sigma*sigma)); k = k/k.sum()
    return reflect_filter(x, k[:, None]*k[None, :])


def area(x, height, width):
    # Unlike adaptive_avg_pool, this integrates fractional pixel overlaps.
    def weights(src, dst):
        scale = src/dst
        left = torch.arange(dst, device=x.device, dtype=x.dtype)[:, None]*scale
        pixel = torch.arange(src, device=x.device, dtype=x.dtype)[None]
        return (torch.minimum(left+scale, pixel+1)-torch.maximum(left, pixel)).clamp_min(0)/scale
    wy, wx = weights(x.shape[-2], height), weights(x.shape[-1], width)
    return torch.matmul(torch.matmul(wy, x), wx.t())


def resize(x, scale):
    h, w = x.shape[-2:]
    return F.interpolate(area(x, max(2, round(h*scale)), max(2, round(w*scale))),
                         size=(h,w), mode='bicubic', align_corners=False)


def warp(x, matrix):
    """Forward pixel-coordinate homography, with OpenCV reflect101 boundaries."""
    h, w = x.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
    xy = torch.stack((xx, yy, torch.ones_like(xx))).reshape(3,-1)
    src = torch.linalg.inv(matrix) @ xy
    sx, sy = src[0]/src[2], src[1]/src[2]
    def fold(t, n):
        if n == 1: return torch.zeros_like(t)
        t = t.remainder(2*n-2)
        return torch.minimum(t, 2*n-2-t)
    sx, sy = fold(sx, w), fold(sy, h)
    grid = torch.stack((2*(sx+.5)/w-1, 2*(sy+.5)/h-1), -1).reshape(1,h,w,2)
    return F.grid_sample(x, grid.expand(x.shape[0],-1,-1,-1), align_corners=False, padding_mode='border')


def shift(x, dx, dy):
    return warp(x, x.new_tensor([[1,0,dx],[0,1,dy],[0,0,1]]))


def ycc(x):
    r,g,b = x[:,0:1],x[:,1:2],x[:,2:3]
    y=.299*r+.587*g+.114*b
    return torch.cat((y, (r-y)*.713+128/255, (b-y)*.564+128/255),1)


def rgb(x):
    y,cr,cb=x[:,0:1],x[:,1:2]-128/255,x[:,2:3]-128/255
    return torch.cat((y+1.403*cr,y-.714*cr-.344*cb,y+1.773*cb),1)


def linear(x):
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055).pow(2.4))


def srgb(x):
    x=x.clamp(0,1)
    return torch.where(x<=.0031308,x*12.92,1.055*x.pow(1/2.4)-.055)


def apply(x, name, generator, params=None, random_inputs=None, jpeg=None):
    if name == 'c3':
        from realesrgan.lenovo_c3 import apply as apply_c3
        if random_inputs:
            raise ValueError('C3 v2 uses multiscale Torch fields; legacy DoG random_inputs are invalid')
        return apply_c3(x, generator, params)
    p=params or {}; noise=random_inputs or {}; x=quantize(x)
    h,w=x.shape[-2:]
    def field(key, channels=1):
        return noise[key] if key in noise else torch.randn((x.shape[0],channels,h,w), device=x.device, dtype=x.dtype, generator=generator)
    def band(z, lo, hi):
        z=gaussian(z,lo)-gaussian(z,hi)
        return z/z.std(dim=(-2,-1),keepdim=True,unbiased=False).clamp_min(1e-6)
    def codec(z,q):
        if jpeg is None: raise ValueError('Torch JPEG callback required; scalar quantization is forbidden')
        return quantize(jpeg(z.clamp(0,1),q))
    meta=dict(p)
    if name=='gaussian':
        result=gaussian(x,p.get('sigma',1.55),True)
    elif name=='defocus':
        r=int(p.get('radius',3)); a=torch.arange(-r,r+1,device=x.device,dtype=x.dtype)
        yy,xx=torch.meshgrid(a,a,indexing='ij'); k=(xx*xx+yy*yy<=r*r).to(x.dtype)
        result=reflect_filter(x,k/k.sum())
    elif name=='motion':
        length=int(p.get('length',11)); angle=p.get('angle',80.)
        k=x.new_zeros((1,1,length,length)); k[:,:,length//2,:]=1/length
        t=math.radians(angle); c,s=math.cos(t),math.sin(t); center=length//2
        matrix=x.new_tensor([[c,s,(1-c)*center-s*center],[-s,c,s*center+(1-c)*center],[0,0,1]])
        # Kernel rotation uses zero exterior, as OpenCV warpAffine does.
        yy,xx=torch.meshgrid(torch.arange(length,device=x.device,dtype=x.dtype),torch.arange(length,device=x.device,dtype=x.dtype),indexing='ij')
        src=torch.linalg.inv(matrix)@torch.stack((xx,yy,torch.ones_like(xx))).reshape(3,-1)
        grid=torch.stack((2*(src[0]+.5)/length-1,2*(src[1]+.5)/length-1),-1).reshape(1,length,length,2)
        k=F.grid_sample(k,grid,align_corners=False)[0,0]; result=reflect_filter(x,k/k.sum())
        meta['angle']=angle
    elif name=='zoom':
        small=quantize(area(x,max(2,round(h*p.get('source_scale',.52))),max(2,round(w*p.get('source_scale',.52)))))
        result=F.interpolate(small,size=(h,w),mode='bicubic',align_corners=False)
    elif name=='c2':
        phase=p.get('phase_xy')
        if phase is None:
            phase=(torch.rand(2,device=x.device,generator=generator)*2-1)*p.get('phase_px',.42)
        z=gaussian(linear(x),p.get('optical_psf_sigma',.82))
        result=srgb(resize(shift(z,phase[0],phase[1]),p.get('sample_scale',.56)))
        meta.update(linear_light=True,optical_psf_sigma=p.get('optical_psf_sigma',.82),reconstruction='bicubic')
    elif name=='c4':
        z=quantize(ycc(x)); z[:,1:]=quantize(gaussian(z[:,1:],p.get('uv_blur_sigma',4),True))
        z=quantize(ycc(quantize(rgb(z))))
        direction=noise.get('direction')
        if direction is None: direction=torch.randn((1,2,1,1),device=x.device,dtype=x.dtype,generator=generator)
        direction=direction/direction.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
        z[:,1:]+=band(field('field'),1.2,5)*direction*p.get('midfreq_strength',3)/255
        result=rgb(z.clamp(0,1).mul(255).floor()/255)
    elif name=='c5':
        result=codec(quantize(x+p.get('usm_amount',1.55)*(x-quantize(gaussian(x,p.get('usm_sigma',2.3),True)))) ,p.get('jpeg_quality',82))
    elif name=='c7':
        z=quantize(shift(x,p.get('shift_x',2.6),p.get('shift_y',1.2)))
        weight=p.get('weight',.35); result=(1-weight)*x+weight*z
    elif name=='c8':
        scale=p.get('display_scale',.62)
        z=codec(quantize(area(x,max(2,round(h*scale)),max(2,round(w*scale)))),p.get('display_jpeg_quality',90))
        z=quantize(F.interpolate(z,size=(h,w),mode='bicubic',align_corners=False))
        src=x.new_tensor([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        offsets=noise.get('corners')
        if offsets is None: offsets=(torch.rand((4,2),device=x.device,generator=generator)*2-1)*p.get('perspective_fraction',.008)*x.new_tensor([w,h])
        dst=src+offsets; rows=[]; values=[]
        for i in range(4):
            a,b=src[i]; u,v=dst[i]; zero=a*0; one=zero+1
            rows.extend([torch.stack([a,b,one,zero,zero,zero,-u*a,-u*b]),torch.stack([zero,zero,zero,a,b,one,-v*a,-v*b])]); values.extend([u,v])
        matrix=torch.cat([torch.linalg.solve(torch.stack(rows),torch.stack(values)),x.new_ones(1)]).reshape(3,3)
        z=quantize(warp(z,matrix)); result=codec(quantize(gaussian(z,p.get('camera_blur_sigma',.7),True)),p.get('final_jpeg_quality',86))
    else:
        raise ValueError('Unknown or pinned atom: '+name)
    return quantize(result),meta
