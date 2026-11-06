import numpy as np
import random
import torch
from torch.nn import functional as F
from tqdm import tqdm
import os.path as osp
import cv2
from collections import OrderedDict
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp, imwrite, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from realesrgan.utils import YUV2RGB, RGB2YUV, YUV2RGBConv, nv21_to_yuv444
import editdistance

@MODEL_REGISTRY.register()
class RealESRNetYUVModel(SRModel):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRNetYUVModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)
        
        self.color = opt.get('color', 'rgb')
        print(f'###################### COLOR is {self.color} ######################')
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.cvtColor = YUV2RGBConv().to(self.device)
        self.finetune_deg = opt.get('finetune_deg', None)

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
        # for paired training or validation
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'mask' in data and data['mask'] is not None:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask = None
        if 'lq_rgb' in data:
            self.lq_rgb = data['lq_rgb'].to(self.device)
        if 'gt_rgb' in data:
            self.gt_rgb = data['gt_rgb'].to(self.device)

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
        
        lq_h, lq_w = self.lq.shape[2:]
        down_rate = self.opt['datasets']['val']['down_rate']
        scale = self.opt.get('scale', 1)
        # if self.color in ('yuv', 'nv12', 'nv21'):
        #     down_rate_h = down_rate*1.5
        #     down_rate_w = down_rate
        # else:
        #     down_rate_h = down_rate_w = down_rate
        down_rate_h = down_rate_w = down_rate
        pad_h = int(np.ceil(lq_h / down_rate_h) * down_rate_h - lq_h)
        pad_w = int(np.ceil(lq_w / down_rate_w) * down_rate_w - lq_w)
        self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
        self.gt = F.pad(self.gt, pad=(0, pad_w*scale, 0, pad_h*scale), mode='reflect')
        if self.color in ('yuv', 'nv12', 'nv21'):
            self.gt_rgb = self.gt
            self.lq_rgb = self.lq
            self.lq = RGB2YUV(self.lq, out_format=self.color)
            self.gt = RGB2YUV(self.gt, out_format=self.color)
    
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

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data_val(val_data)
            
            self.test()

            # print("Total:", )

            # cv2.imwrite("mask_" + str(idx) + ".png", self.mask.data.cpu().numpy().astype(np.uint8)[0][0] * 255)

            if self.color in ('yuv', 'nv12', 'nv21'):
                lq_h, lq_w = self.lq_rgb.shape[2:]
            else:
                lq_h, lq_w = self.lq.shape[2:]
            scale = self.opt.get('scale', 1)
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
        if self.color in ('yuv', 'nv12', 'nv21'):
            # out_dict['lq'] = YUV2RGB(self.lq, format='RGB').detach().cpu()
            # out_dict['result'] = YUV2RGB(self.output, format='RGB').detach().cpu()
            # if hasattr(self, 'gt'):
            #     out_dict['gt'] = YUV2RGB(self.gt, format='RGB').detach().cpu()
            out_dict['lq'] = self.cvtColor(self.lq_rgb).detach().cpu()
            out_dict['result'] = self.cvtColor(self.output).detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt_rgb.detach().cpu()
        else:
            out_dict['lq'] = self.lq.detach().cpu()
            out_dict['result'] = self.output.detach().cpu()
            if hasattr(self, 'gt'):
                out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'gt_lines'):
            out_dict['gt_lines'] = self.gt_lines
        return out_dict
    
    def optimize_parameters(self, current_iter):
        # if current_iter % 2500 == 0:
        #     for i in range(self.lq.size()[0]):
        #         img_show = (self.lq[i][:3, ...].detach().cpu().permute(1, 2, 0).clamp(min=0, max=1).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        #         # img_show = cv2.resize(img_show, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        #         cv2.imwrite(f'datasets/debug_train_vis/lq_train_{current_iter}_{i}.png', img_show)
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        # if current_iter % 2500 == 0:
        #     for i in range(self.lq.size()[0]):
        #         img_show = (self.output[i][:3, ...].detach().cpu().permute(1, 2, 0).clamp(min=0, max=1).numpy() * 255).astype(np.uint8)[:, :, ::-1]
        #         # img_show = cv2.resize(img_show, dsize=(200, 200), interpolation=cv2.INTER_CUBIC)
        #         cv2.imwrite(f'datasets/debug_train_vis/lq_train_{current_iter}_{i}_out.png', img_show)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if self.color in ('yuv', 'nv12', 'nv21'):
                if self.opt.get('l1', 'rgb') == 'yuv':
                    l_pix = self.cri_pix(self.output, RGB2YUV(self.gt_rgb, out_format='yuv'), mask=self.mask)
                else:
                    l_pix = self.cri_pix(self.cvtColor(self.output), self.gt_rgb, mask=self.mask)
            else:
                l_pix = self.cri_pix(self.output, self.gt, mask=self.mask)
            # l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        if self.cri_rec:
            if current_iter < 25000:
                self.cri_rec.loss_weight = 1.0
            else:
                self.cri_rec.loss_weight = self.opt['train']['rec_opt']['loss_weight']
            if self.color in ('yuv', 'nv12', 'nv21'):
                l_rec = self.cri_rec(self.cvtColor(self.output), self.gt_rgb)
            else:
                l_rec = self.cri_rec(self.output, self.gt)
            if current_iter % 200 == 1:
                if self.color in ('yuv', 'nv12', 'nv21'):
                    out_show = tensor2img(self.cvtColor(self.output))
                else:
                    out_show = tensor2img(self.output)
                cv2.imwrite(f'./datasets/train_vis_nv21/out_show_{current_iter-1}.png', out_show)
            l_total += l_rec
            loss_dict['l_rec'] = l_rec
        
        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

