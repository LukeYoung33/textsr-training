import numpy as np
import random
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os.path as osp

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp, tensor2img, imwrite
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from basicsr.metrics import calculate_metric
from realesrgan.utils import YUV2RGB, RGB2YUV
import editdistance


@MODEL_REGISTRY.register()
class RealESRGANModel(SRGANModel):
    """RealESRGAN Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        
        self.color = opt.get('color', 'rgb')
        print(f'###################### COLOR is {self.color} ######################')

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt_usm = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(self.gt_usm, self.kernel1)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
        if self.color == 'yuv':
            self.lq = RGB2YUV(self.lq)
            self.gt = RGB2YUV(self.gt)

    def feed_data_val(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'label_mask' in data:
            self.label_mask = data['label_mask'].to(self.device).float()
        if 'label_mask_gt' in data:
            self.label_mask = data['label_mask_gt'].to(self.device).float()
        if 'gt_lines' in data:
            self.gt_lines = data['gt_lines']
        if self.color == 'yuv':
            self.lq = RGB2YUV(self.lq)
            self.gt = RGB2YUV(self.gt)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        save_recognition = self.opt['val'].get('save_recognition', False)
        comparison = self.opt['val'].get('comparison', None)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if "recognition" in self.opt['val']['metrics'].keys():
            self.sr_rec_list = []
            self.gt_rec_list = []
        if "recognition_divide" in self.opt['val']['metrics'].keys():
            self.sr_rec = {
                "CHN":[],
                "ENG":[]
            }
            self.gt_rec = {
                "CHN": [],
                "ENG": []
            }

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        down_rate = self.opt['val'].get('down_rate', 1)
        scale = self.opt.get('scale', 1)

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_val(val_data)
            lq_h, lq_w = self.lq.shape[2:]
            pad_h = int(np.ceil(lq_h / down_rate) * down_rate - lq_h)
            pad_w = int(np.ceil(lq_w / down_rate) * down_rate - lq_w)
            self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
            self.gt = F.pad(self.gt, pad=(0, pad_w*scale, 0, pad_h*scale), mode='reflect')
            self.test()

            # print("Total:", )

            # cv2.imwrite("mask_" + str(idx) + ".png", self.mask.data.cpu().numpy().astype(np.uint8)[0][0] * 255)

            visuals = self.get_current_visuals()
            self.lq = self.lq[..., :lq_h, :lq_w]
            self.gt = self.gt[..., :lq_h * scale, :lq_w * scale]
            visuals['result'] = visuals['result'][..., :lq_h * scale, :lq_w * scale]
            visuals['lq'] = visuals['lq'][..., :lq_h, :lq_w]

            sr_img = tensor2img([visuals['result']])
            H, W = sr_img.shape[:2]
            img_bicubic = torch.nn.functional.interpolate(visuals['lq'], (H, W), mode="bicubic")
            lr_img = tensor2img([img_bicubic])
            #sr_img
            if 'gt' in visuals:
                visuals['gt'] = visuals['gt'][..., :lq_h * scale, :lq_w * scale]
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img[..., :3]
                del self.gt
            if 'mask' in visuals:
                # print("mask:", mask.shape)
                mask = tensor2img([visuals['mask']])
                metric_data['mask'] = mask
            if 'gt_lines' in visuals:
                # print("mask:", mask.shape)
                # mask = tensor2img([visuals['gt_lines']])
                metric_data['gt_lines'] = visuals['gt_lines']

            if comparison == "LR":
                sr_img = lr_img
            elif comparison == "HR":
                sr_img = gt_img
            metric_data['img'] = sr_img[..., :3] ###sr_img
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
            if save_recognition:
                if self.opt['is_train']:
                    save_rec_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.txt')
                else:
                    if self.opt['val']['suffix']:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.txt')
                    else:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.txt')

            if with_metrics:
                # calculate metrics
                rec_cnt = 0
                # print("metrics:", self.opt['val']['metrics'].keys())
                for name, opt_ in self.opt['val']['metrics'].items():
                    if "recognition" in name:
                        # recognition performs only once
                        if rec_cnt > 0:
                            continue
                        
                        sr_rec, gt_str = calculate_metric(metric_data, opt_)
                        rec_cnt += 1
                        if "recognition" in self.opt['val']['metrics'].keys():
                            if type(sr_rec) == dict:
                                for key in sr_rec:
                                    self.sr_rec_list.extend(sr_rec[key])
                                    self.gt_rec_list.extend(gt_str[key])

                                    if save_recognition:
                                        rec_f = open(save_rec_path, "a+")
                                        for i in range(len(sr_rec[key])):
                                            sr_, gt_ = sr_rec[key][i], gt_str[key][i]
                                            rec_f.write(sr_ + "\t" + gt_ + "\n")
                                        rec_f.close()
                            else:
                                self.sr_rec_list.extend(sr_rec)
                                self.gt_rec_list.extend(gt_str)
                        if "recognition_divide" in self.opt['val']['metrics'].keys():
                            # sr_rec, gt_str = calculate_metric(metric_data, opt_)
                            for key in sr_rec:
                                self.sr_rec[key].extend(sr_rec[key])
                                self.gt_rec[key].extend(gt_str[key])
                    elif "ned" in name:
                        pass
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if metric == "recognition":
                    sr_list = self.sr_rec_list
                    gt_list = self.gt_rec_list
                    cnt = 0
                    for i in range(len(sr_list)):
                        if sr_list[i] == gt_list[i]:
                            cnt += 1
                        if "ned" in self.metric_results.keys():
                            pred, gt = sr_list[i], gt_list[i]
                            max_len = max(len(pred), len(gt))
                            self.metric_results["ned"] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                    self.metric_results[metric] = cnt / float(len(sr_list) + 1e-10)
                    if "ned" in self.metric_results.keys():
                        self.metric_results["ned"] /= float(len(sr_list) + 1e-10)
                    # self.metric_results

                    self.sr_rec_list = []
                    self.gt_rec_list = []

                elif metric == "recognition_divide":
                    sr_list = self.sr_rec
                    gt_list = self.gt_rec
                    cnt = {
                        "CHN": 0,
                        "ENG": 0
                    }
                    for key in sr_list:
                        for i in range(len(sr_list[key])):
                            if sr_list[key][i] == gt_list[key][i]:
                                cnt[key] += 1
                            if ("ned_" + key) in self.metric_results.keys():
                                pred, gt = sr_list[key][i], gt_list[key][i]
                                max_len = max(len(pred), len(gt))
                                self.metric_results["ned_" + key] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                        self.metric_results["recognition_" + key] = cnt[key] / float(len(sr_list[key]) + 1e-10)
                        if "ned_" + key in self.metric_results.keys():
                            self.metric_results["ned_" + key] /= float(len(sr_list[key]) + 1e-10)
                        # self.metric_results
                        # print("cnt", cnt, len(sr_list[key]))
                    self.sr_rec = {
                        "CHN": [],
                        "ENG": []
                    }
                    self.gt_rec = {
                        "CHN": [],
                        "ENG": []
                    }

                elif "ned" in metric:
                    pass
                elif "recognition" in metric:
                    pass
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            
            self.is_train = True

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.color == 'yuv':
            out_dict['lq'] = YUV2RGB(self.lq, format='RGB').detach().cpu()
            out_dict['result'] = YUV2RGB(self.output, format='RGB').detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = YUV2RGB(self.gt, format='RGB').detach().cpu()
        else:
            out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'gt_lines'):
            out_dict['gt_lines'] = self.gt_lines
        return out_dict
    
    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_g_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                if self.color == 'yuv':
                    l_g_percep, l_g_style = self.cri_perceptual(YUV2RGB(self.output), YUV2RGB(percep_gt))
                else:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            if self.opt['train']['gan_opt']['type'] in ('RelativisticSoftplusGANLoss'):
                target = self.net_d(gan_gt).detach().clone()
                l_g_gan = self.cri_gan(fake_g_pred, target, is_disc=False)
            else:
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
            if self.cri_rec:
                if self.color == 'yuv':
                    l_g_rec = self.cri_rec(YUV2RGB(self.output), YUV2RGB(self.gt))
                else:
                    l_g_rec = self.cri_rec(self.output, self.gt)
                l_total += l_g_rec
                loss_dict['l_g_rec'] = l_g_rec
            
            # if self.cri_grad:
            #     l_g_grad = self.cri_grad(self.output, self.gt)
            #     l_total += l_g_grad
            #     loss_dict['l_grad'] = l_g_grad

            l_g_total.backward()
            self.optimizer_g.step()

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            if self.opt['train']['gan_opt']['type'] in ('RelativisticSoftplusGANLoss'):
                real_d_pred = self.net_d(gan_gt)
                fake_d_pred = self.net_d(self.output.detach().clone())
                l_d_real = self.cri_gan(fake_d_pred, real_d_pred, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                l_d_real.backward()
            else:
                # real
                real_d_pred = self.net_d(gan_gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                l_d_real.backward()
                # fake
                fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_fake.backward()
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
