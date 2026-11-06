import numpy as np
import os
from easydict import EasyDict
import realesrgan.metrics.crnn as crnn
import torch
from torch.autograd import Variable
from basicsr.utils.registry import METRIC_REGISTRY
import cv2
import string
import math
from pathlib import Path
# import levensteindistance
import pdb
from realesrgan.archs.ppocrv5_arch import PPOCRv5MobileRec, decode_ctc

global_device="cpu"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PPOCRV5_DEFAULT_DICT = _REPO_ROOT / "assets" / "ppocrv5_dict.txt"
_PPOCRV5_DEFAULT_WEIGHTS = _REPO_ROOT / "weights" / "ptocr_v5_mobile_rec.pth"
_ppocrv5_recognizer = None
_ppocrv5_recognizer_device = None
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

def str_filt(str_, voc_type="benchmark"):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation,
        'chinese': open("realesrgan/metrics/al_chinese.txt", "r", encoding='utf-8').readlines()[0].replace("\n", ""),
        'benchmark': open("realesrgan/metrics/benchmark.txt", "r", encoding='utf-8').readlines()[0].replace("\n", "")
    }
    if voc_type == 'lower':
        str_ = str_.lower()

    if voc_type in ['chinese', 'benchmark']: # Chinese character only
        new_str = ""
        for ch in str_:
            # if '\u4e00' <= ch <= '\u9fa5' or ch in string.digits + string.ascii_letters:
            new_str += ch
        str_ = new_str
    if type(str_) == list:
        str_ = str_[0]
    for char in str_:
        if char not in alpha_dict[voc_type]: #voc_type
            str_ = str_.replace(char, '')
    return str_

def get_string_crnn(outputs_, use_chinese, alphabet='-0123456789abcdefghijklmnopqrstuvwxyz'):
    outputs = outputs_.permute(1, 0, 2).contiguous()
    predict_result = []

    if use_chinese:
        # alphabet = open("basicsr/metrics/al_chinese.txt", 'r').readlines()[0].replace("\n", "")
        alphabet = list(open("realesrgan/metrics/benchmark.txt", "r", encoding='utf-8').readlines()[0].replace("\n", ""))
        alphabet.insert(0, ' ')

    for output in outputs:
        max_index = torch.max(output, 1)[1]

        out_str = ""
        last = ""
        for i in max_index:
            if alphabet[i] != last:
                if i != 0:
                    out_str += alphabet[i]
                    last = alphabet[i]
                else:
                    last = ""

        predict_result.append(out_str)
    return predict_result

def CRNN_init(recognizer_path=None, opt=None):

    alphabet = open("realesrgan/metrics/benchmark.txt", 'r', encoding='utf-8').readlines()[0].replace("\n", "")

    model = crnn.CRNN(3, 256, len(alphabet) + 1, 32)
    model = model.to(global_device)
    # model.eval()
    # cfg = self.config.TRAIN
    # aster_info = AsterInfo(cfg.voc_type)
    model_path = recognizer_path if not recognizer_path is None else opt.saved_model
    print('loading pretrained TPG model from %s' % model_path)
    stat_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(stat_dict)

    load_keys = stat_dict.keys()
    man_load_dict = model.state_dict()
    for key in stat_dict:
        if not key.replace("module.", "") in man_load_dict:
            print("Key not match", key, key.replace("module.", ""))
        man_load_dict[key.replace("module.", "")] = stat_dict[key]
    model.load_state_dict(man_load_dict)


    return model

def CRNN_ENG_init(recognizer_path=None, opt=None):

    model = crnn.CRNN_ENG(32, 1, 37, 256)
    model = model.to(global_device)
    # cfg = self.config.TRAIN
    # aster_info = AsterInfo(cfg.voc_type)
    model_path = recognizer_path if not recognizer_path is None else opt.saved_model_eng
    print('loading pretrained TPG model from %s' % model_path)
    stat_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model_keys = model.state_dict().keys()
    # print("state_dict:", len(stat_dict))
    if type(stat_dict) == list:
        print("state_dict:", len(stat_dict))
        stat_dict = stat_dict[0]  # .state_dict()
    # load_keys = stat_dict.keys()

    if recognizer_path is None:
        # model.load_state_dict(stat_dict)
        load_keys = stat_dict.keys()
        man_load_dict = model.state_dict()
        for key in stat_dict:
            if not key.replace("module.", "") in man_load_dict:
                print("Key not match", key, key.replace("module.", ""))
            man_load_dict[key.replace("module.", "")] = stat_dict[key]
        model.load_state_dict(man_load_dict)
    else:
        # model = stat_dict
        model.load_state_dict(stat_dict)
    return model

print("Initializing recognizer...")
recognizer = CRNN_init(opt=opt)
print("done")

recognizers = {
    "CHN":recognizer,
    "ENG":None
}


def get_recognizer(lang):
    if recognizers[lang] is None:
        if lang == "ENG":
            recognizers[lang] = CRNN_ENG_init(opt=opt)
        else:
            recognizers[lang] = CRNN_init(opt=opt)
    return recognizers[lang]


def get_ppocrv5_recognizer(dict_path=None, weights_path=None, device=None, use_space_char=True):
    global _ppocrv5_recognizer, _ppocrv5_recognizer_device

    dict_path = str(Path(dict_path) if dict_path is not None else _PPOCRV5_DEFAULT_DICT)
    weights_path = str(Path(weights_path) if weights_path is not None else _PPOCRV5_DEFAULT_WEIGHTS)
    device = device or global_device

    if _ppocrv5_recognizer is None:
        model = PPOCRv5MobileRec(
            character_dict_path=dict_path,
            use_space_char=use_space_char
        )
        model.load_weights(weights_path, map_location="cpu", strict=True)
        model.eval()
        _ppocrv5_recognizer = model
        _ppocrv5_recognizer_device = None

    if _ppocrv5_recognizer_device != device:
        _ppocrv5_recognizer = _ppocrv5_recognizer.to(device)
        _ppocrv5_recognizer_device = device

    return _ppocrv5_recognizer


def parse_ppocrv5_data(img, crop_height=48, min_width=16, max_width=512):
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        return None

    height, width = img.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = float(crop_height) / float(height)
    target_width = max(min_width, min(int(round(width * scale)), max_width))
    resized = cv2.resize(img, (target_width, crop_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def recognition_processing_ppocrv5(
    img,
    eval=True,
    dict_path=None,
    weights_path=None,
    use_space_char=True,
    crop_height=48,
    min_width=16,
    max_width=512
):
    processed = parse_ppocrv5_data(
        img,
        crop_height=crop_height,
        min_width=min_width,
        max_width=max_width
    )
    if processed is None:
        return ""

    recognizer_model = get_ppocrv5_recognizer(
        dict_path=dict_path,
        weights_path=weights_path,
        device=global_device,
        use_space_char=use_space_char
    )
    with torch.no_grad():
        if eval:
            recognizer_model.eval()
        output = recognizer_model(processed.to(global_device))
    return decode_ctc(
        output,
        character_dict_path=str(Path(dict_path) if dict_path is not None else _PPOCRV5_DEFAULT_DICT),
        use_space_char=use_space_char
    )[0][0]


def normalize_recognition_text(text, lang, voc_types):
    if lang == "ENG":
        text = text.lower()
    return str_filt(text, voc_types[lang]).replace(" ", "")

def parse_ENG_data(imgs_input_, ratio_keep=True):

    in_width = 512

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic') / float(255)

    # print("imgs_input:", imgs_input.shape)

    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def parse_CRNN_data(imgs_input_, ratio_keep=True):

    in_width = 512

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

    return imgs_input

data_parser = {
    "CHN":parse_CRNN_data,
    "ENG":parse_ENG_data
}


def recognition_loss(img_sr, img_hr, lang="CHN", eval=False):
    # print("lang:", lang)
    # im_sr = img_sr
    # im_sr_th = torch.tensor(im_sr).permute(2, 0, 1).unsqueeze(0).float()
    processed_sr = data_parser[lang](img_sr)
    processed_hr = data_parser[lang](img_hr)
    # with torch.no_grad():
    # if eval and lang=="ENG":
    # recognizers[lang].eval()
    recognizer_model = get_recognizer(lang)
    sr_output = recognizer_model(processed_sr.to(global_device))
    hr_output = recognizer_model(processed_hr.to(global_device))
    print("sr_output:", sr_output.shape)
    # if eval and lang=="ENG":
    # recognizers[lang].train()
    # sr_prediction = get_string_crnn(sr_output, lang==
    sr_log = sr_output.log_softmax(dim=1)
    hr_log = hr_output.log_softmax(dim=1)
    hr_prob = hr_output.softmax(dim=1)
    return hr_prob * (hr_log - sr_log)


def recognition_processing(img, lang="CHN", eval=False):
    # print("lang:", lang)
    im_sr = img
    im_sr_th = torch.tensor(im_sr).permute(2, 0, 1).unsqueeze(0).float()
    processed_sr = data_parser[lang](im_sr_th)
    with torch.no_grad():
        if eval and lang=="ENG":
            get_recognizer(lang).eval()
        recognizer_model = get_recognizer(lang)
        sr_output = recognizer_model(processed_sr.to(global_device))
        if eval and lang=="ENG":
            recognizer_model.train()
    sr_prediction = get_string_crnn(sr_output, lang=="CHN")
    return sr_prediction


def get_polys(annos):
    if annos and isinstance(annos[0], list):
        flat_annos = []
        for anno_group in annos:
            if isinstance(anno_group, list):
                flat_annos.extend(anno_group)
            else:
                flat_annos.append(anno_group)
        annos = flat_annos
    polys = np.array([np.array(anno.split(",")[:8]).astype(np.int_) for anno in annos]).reshape(-1, 4, 2)
    gt_strs = [(",".join((anno.split(",")[8:]))).replace("\n", "") for anno in annos]
    return polys, gt_strs

debug_visdir = "cropped_ims"
if not os.path.isdir(debug_visdir):
    os.makedirs(debug_visdir)

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


def get_lang(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return "CHN"
    return "ENG"


@METRIC_REGISTRY.register()
def calculate_recognition(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = []
    gt_strs_filt = []
    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang)
        sr_predictions.append(str_filt(rec, voc_types[lang]))
        gt_str = str_filt(gt_strs[cnt], voc_types[all_langs[cnt]]) # all to be lower
        gt_strs_filt.append(gt_str)
        print(str_filt(rec, voc_types[lang]), "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt


@METRIC_REGISTRY.register()
def calculate_recognition_case_insensitive(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = []
    gt_strs_filt = []
    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang, eval=True)
        sr_predictions.append(str_filt(rec, voc_types[lang]).replace(" ", ""))
        gt_str = str_filt(gt_strs[cnt].lower(), voc_types[all_langs[cnt]]).replace(" ", "") # all to be lower
        gt_strs_filt.append(gt_str)
        print(str_filt(rec, voc_types[lang]), "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt

@METRIC_REGISTRY.register()
def calculate_recognition_case_insensitive_langdivide(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = {
        "ENG": [],
        "CHN": []
    }

    gt_strs_filt = {
        "ENG": [],
        "CHN": []
    }

    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang, eval=True)
        pred_str = str_filt(rec, voc_types[lang]).replace(" ", "")
        sr_predictions[lang].append(pred_str)
        gt_str = str_filt(gt_strs[cnt].lower(), voc_types[all_langs[cnt]]).replace(" ", "") # all to be lower
        gt_strs_filt[lang].append(gt_str)
        # print(pred_str, "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt


@METRIC_REGISTRY.register()
def calculate_recognition_case_insensitive_langdivide_ppocrv5(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = {
        "ENG": [],
        "CHN": []
    }

    gt_strs_filt = {
        "ENG": [],
        "CHN": []
    }

    dict_path = kwargs.get("dict_path", str(_PPOCRV5_DEFAULT_DICT))
    weights_path = kwargs.get("weights_path", str(_PPOCRV5_DEFAULT_WEIGHTS))
    use_space_char = kwargs.get("use_space_char", True)
    crop_height = kwargs.get("crop_height", 48)
    min_width = kwargs.get("min_width", 16)
    max_width = kwargs.get("max_width", 512)

    for cnt, im in enumerate(sr_warpped):
        lang = get_lang(gt_strs[cnt])
        rec = recognition_processing_ppocrv5(
            im,
            eval=True,
            dict_path=dict_path,
            weights_path=weights_path,
            use_space_char=use_space_char,
            crop_height=crop_height,
            min_width=min_width,
            max_width=max_width
        )
        pred_str = normalize_recognition_text(rec, lang, voc_types)
        gt_str = normalize_recognition_text(gt_strs[cnt], lang, voc_types)
        sr_predictions[lang].append(pred_str)
        gt_strs_filt[lang].append(gt_str)

    return sr_predictions, gt_strs_filt


@METRIC_REGISTRY.register()
def calculate_recognition_case_relative_langdivide(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)
    gt_warpped = crop_images(img2, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = {
        "ENG": [],
        "CHN": []
    }

    gt_strs_filt = {
        "ENG": [],
        "CHN": []
    }

    # cnt = 0
    all_langs = []
    for cnt, im in enumerate(sr_warpped):
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        sr_rec = recognition_processing(im, lang, eval=True)
        pred_str = str_filt(sr_rec, voc_types[lang]).replace(" ", "")
        sr_predictions[lang].append(pred_str)
        # gt_str = str_filt(gt_strs[cnt].lower(), voc_types[all_langs[cnt]]).replace(" ", "") # all to be lower
        # gt_strs_filt[lang].append(gt_str)
        gt_rec = recognition_processing(gt_warpped[cnt], lang, eval=True)
        gt_str = str_filt(gt_rec, voc_types[lang]).replace(" ", "")
        gt_strs_filt[lang].append(gt_str)
        # print(pred_str, "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt


if __name__=="__main__":
    import cv2
    import torch.nn as nn
    from basicsr.utils import tensor2img

    img_gt = 'datasets/debug/gt_25.png'
    img_lr = 'datasets/debug/lq_n6_b2_25.png'
    device = global_device

    img_gt = (torch.from_numpy(cv2.imread(img_gt)).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)
    img_lr = (torch.from_numpy(cv2.imread(img_lr)).to(torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

    model = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False).to(device)

    img_sr = model(img_lr)

    img_gt = tensor2img(img_gt)
    img_sr = tensor2img(img_sr)

    rec = calculate_recognition_case_insensitive_langdivide()

