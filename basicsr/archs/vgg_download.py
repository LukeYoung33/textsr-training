from torchvision.models import vgg as vgg

vgg_type = 'vgg19'
vgg_net = getattr(vgg, vgg_type)(pretrained=True)
