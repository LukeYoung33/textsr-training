import torch
import torch.nn as nn
from easydict import EasyDict
import sys
sys.path.append('.')
from basicsr.utils.registry import LOSS_REGISTRY
import realesrgan.metrics.crnn as crnn
import numpy as np

opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "weights/scene_base_CRNN.pth",#"best_accuracy.pth", #"None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "saved_model_eng": "weights/crnn.pth",
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz",
        "character_eng": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

# if args.CHNSR:
opt['character'] = open("realesrgan/metrics/al_chinese.txt", 'r', encoding='utf-8').readlines()[0].replace("\n", "")
opt["num_class"] = len(opt['character'])
opt = EasyDict(opt)
global_device="cuda:0"

def CRNN_init(recognizer_path=None, opt=None):

    alphabet = open("realesrgan/metrics/benchmark.txt", 'r', encoding='utf-8').readlines()[0].replace("\n", "")

    model = crnn.CRNN(3, 256, len(alphabet) + 1, 32)
    model = model.to(global_device)
    # cfg = self.config.TRAIN
    # aster_info = AsterInfo(cfg.voc_type)
    model_path = recognizer_path if not recognizer_path is None else opt.saved_model
    print('loading pretrained TPG model from %s' % model_path)
    stat_dict = torch.load(model_path)
    # model.load_state_dict(stat_dict)

    load_keys = stat_dict.keys()
    man_load_dict = model.state_dict()
    for key in stat_dict:
        if not key.replace("module.", "") in man_load_dict:
            print("Key not match", key, key.replace("module.", ""))
        man_load_dict[key.replace("module.", "")] = stat_dict[key]
    model.load_state_dict(man_load_dict)

    return model


def parse_CRNN_data(imgs_input_, ratio_keep=True):

    in_width = 128

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate((imgs_input_ * 255.0), (32, in_width), mode='bicubic')

    return imgs_input


def parse_CRNN_dataV2(imgs_input_, gt_lines, ratio_keep=True):
    imgs_input_list = []
    in_width = 128
    for i in range(imgs_input_.size()[0]):
        imgs_input = imgs_input_[i:i+1, ...]
        for line in gt_lines[i]:
            coord = np.array(list(map(int, line.split(',')[:8])))
            if not len(coord) == 8:
                print(coord)
                continue
            box = [coord[0::2].min(), coord[1::2].min(), coord[0::2].max(), coord[1::2].max()]
            cropped_img = imgs_input[..., box[1]:box[3], box[0]:box[2]]

            if ratio_keep:
                real_height, real_width = imgs_input.shape[2:]
                ratio = real_width / float(real_height)

                # if ratio > 3:
                in_width = max(min(int(ratio * 32), 1024), 16)
            imgs_input = torch.nn.functional.interpolate((cropped_img * 255.0), (32, in_width), mode='bicubic')
            imgs_input_list.append(imgs_input)

    return torch.cat(imgs_input_list, dim=0)

@LOSS_REGISTRY.register()
class RecognitionLoss(nn.Module):
    def __init__(self, loss_weight=1e-4, reduction='mean'):
        super(RecognitionLoss, self).__init__()
        self.func = nn.L1Loss(reduce=False)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction

        self.recognizer = CRNN_init(opt=opt)

    def forward(self, out_images, target_images, mask=None):
        """
        out_images: B, 3, H, W
        target_images: B, 3, H, W
        mask: B, 1, H, W
        """

        out_images_parsed = parse_CRNN_data(out_images)
        target_images_parsed = parse_CRNN_data(target_images)

        # print("weight_mask:", out_images_parsed.shape, target_images_parsed.shape)
        # pdb.set_trace()
        out_images_parsed = out_images_parsed[:, :3, ...] if out_images_parsed.shape[1]>3 else out_images_parsed
        target_images_parsed = target_images_parsed[:, :3, ...] if target_images_parsed.shape[1]>3 else target_images_parsed

        out_logits = self.recognizer(out_images_parsed)
        target_logits = self.recognizer(target_images_parsed)

        # sr_log = out_logits.log_softmax(dim=-1)
        # hr_log = target_logits.log_softmax(dim=-1)
        hr_prob = target_logits.softmax(dim=-1)

        hr_maxprob_idx = hr_prob.max(dim=-1)[1]
        hr_discrete_label = hr_maxprob_idx #torch.nn.functional.one_hot(, num_classes=hr_prob.shape[-1]).to(hr_prob.device)
        hr_discrete_label = hr_discrete_label.reshape(-1)

        sr_logit = out_logits.reshape(-1, hr_prob.shape[-1])

        loss_rec = torch.nn.functional.cross_entropy(sr_logit, hr_discrete_label.detach())

        # print("hr_prob:", hr_discrete_label.shape, sr_logit.shape, loss_rec)

        # loss_rec = torch.abs(out_logits - target_logits.detach()).mean()
        # print("loss_rec:", loss_rec)
        # weight_mask = (mask > 0).float() * 1.0 + (mask == 0).float() * 0.5
        # print("weight_mask:", loss_rec)

        return loss_rec.mean() * 1e-1

def get_polys(annos):
    # print (annos[0])
    polys = np.array([np.array(anno[0].split(",")[:8]).astype(np.int_) for anno in annos]).reshape(-1, 4, 2)
    gt_strs = [(",".join((anno[0].split(",")[8:]))).replace("\n", "") for anno in annos]
    return polys, gt_strs


def crop_images(global_image, polys):
    # img = cv2.imread("big_vertical_text.jpg")
    # points for test.jpg
    all_warpped = []
    # print("polys:", polys)
    for idx in range(len(polys)):
        img = global_image.copy()
        cnt = polys[idx]
        # print("shape of cnt: {}".format(cnt.shape))
        # print("cnt:", cnt)
        rect = cv2.minAreaRect(cnt)
        #print("rect: {}".format(rect))

        # the order of the box points: bottom left, top left, top right,
        # bottom right

        # get width and height of the detected rectangle
        if abs(rect[2]) > 45:
            width = rect[1][0]
            height = rect[1][1]
            rect = (rect[0], (height, width), -(90 - abs(rect[2]))) ############
        # rect = (rect[0], (rect[1][0], rect[1][1] * 1.1), rect[2])
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int_(box)

        # print("bounding box: {}".format(box))

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))
        all_warpped.append(warped)
        # cv2.waitKey(0)

    return all_warpped


@LOSS_REGISTRY.register()
class RecognitionV2Loss(nn.Module):
    def __init__(self, loss_weight=1e-4, reduction='mean'):
        super(RecognitionV2Loss, self).__init__()
        self.func = nn.L1Loss(reduce=False)
        self.loss_weight = float(loss_weight)
        self.reduction = reduction

        self.recognizer = CRNN_init(opt=opt)

    def forward(self, out_images, target_images, mask=None):

        # out_images_parsed = parse_CRNN_dataV2(out_images)
        # target_images_parsed = parse_CRNN_dataV2(target_images)

        # print("weight_mask:", out_images_parsed.shape, target_images_parsed.shape)
        # pdb.set_trace()
        out_images_parsed = out_images[:, :3, ...]
        target_images_parsed = target_images[:, :3, ...]

        out_feats = self.recognizer.get_feature(out_images_parsed)
        target_feats = self.recognizer.get_feature(target_images_parsed)


        total_loss = 0
        for out_feat, target_feat in zip(out_feats, target_feats):
            loss_rec = torch.nn.functional.l1_loss(out_feat, target_feat.detach())
            # loss_sim = torch.nn.functional.cosine_similarity(out_feat, target_feat.detach()).mean()
            total_loss += loss_rec      # + loss_sim.mean() * 1e-1

        # print("hr_prob:", hr_discrete_label.shape, sr_logit.shape, loss_rec)

        # loss_rec = torch.abs(out_logits - target_logits.detach()).mean()
        # print("loss_rec:", loss_rec)
        # weight_mask = (mask > 0).float() * 1.0 + (mask == 0).float() * 0.5
        # print("weight_mask:", loss_rec)

        return total_loss * self.loss_weight


if __name__=="__main__":
    import cv2

    img_gt = 'datasets/debug/gt_25.png'
    img_lr = 'datasets/debug/lq_n6_b2_25.png'
    device = global_device

    img_gt = (torch.from_numpy(cv2.imread(img_gt)).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)
    img_lr = (torch.from_numpy(cv2.imread(img_lr)).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

    model = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False).to(device)
    cri = RecognitionLoss(loss_weight=1)

    img_sr = model(img_lr)

    rec_loss = cri(img_sr, img_gt)

    print(rec_loss)



