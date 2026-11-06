import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')
from realesrgan.archs.ppocrv5_utils import PPLCNetV3, SequenceEncoder, CTCHead

def _count_classes_from_dict(character_dict_path, use_space_char=True):
    if character_dict_path is None:
        raise ValueError("character_dict_path is required to infer num_classes")
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
    num_chars = len(lines)
    if use_space_char:
        num_chars += 1
    # CTC adds a blank token at index 0
    return num_chars + 1


class PPOCRv5MobileRec(nn.Module):
    """
    PP-OCRv5 mobile recognition model (SVTR_LCNet + CTC) as a standalone nn.Module.
    This mirrors configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml for inference.
    """

    def __init__(
        self,
        num_classes=None,
        character_dict_path=None,
        use_space_char=True,
        backbone_scale=0.95,
        svtr_dims=120,
        svtr_depth=2,
        svtr_hidden_dims=120,
        svtr_kernel_size=(1, 3),
        svtr_use_guide=True,
        ctc_fc_decay=1e-5,
    ):
        super().__init__()

        if num_classes is None:
            num_classes = _count_classes_from_dict(character_dict_path, use_space_char)

        self.num_classes = num_classes

        self.backbone = PPLCNetV3(scale=backbone_scale)
        self.ctc_neck = SequenceEncoder(
            in_channels=self.backbone.out_channels,
            encoder_type="svtr",
            dims=svtr_dims,
            depth=svtr_depth,
            hidden_dims=svtr_hidden_dims,
            kernel_size=list(svtr_kernel_size),
            use_guide=svtr_use_guide,
        )
        self.ctc_head = CTCHead(
            in_channels=self.ctc_neck.out_channels,
            out_channels=self.num_classes,
            fc_decay=ctc_fc_decay,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.ctc_neck(x)
        x = self.ctc_head(x)
        return x

    def load_weights(self, weights_path, map_location="cpu", strict=True):
        state = torch.load(weights_path, map_location=map_location)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            elif "net" in state:
                state = state["net"]

        if any(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}

        # Update the state dict keys to match the current model's keys if necessary
        for k in list(state.keys()):
            if k.startswith("backbone."):
                new_k = k.replace("backbone.", "backbone.")
                state[new_k] = state.pop(k)
            elif k.startswith("head.ctc_encoder."):
                new_k = k.replace("head.ctc_encoder.", "ctc_neck.")
                state[new_k] = state.pop(k)
            elif k.startswith("head.ctc_head."):
                new_k = k.replace("head.ctc_head.", "ctc_head.")
                state[new_k] = state.pop(k)
        incompat = self.load_state_dict(state, strict=strict)
        return {"missing": incompat.missing_keys, "unexpected": incompat.unexpected_keys}


def _preprocess_no_resize_bgr(img_bgr):
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with 3 channels.")
    img = img_bgr.astype("float32")
    img = img.transpose(2, 0, 1) / 255.0
    img -= 0.5
    img /= 0.5
    return img


def decode_ctc(preds, character_dict_path, use_space_char=True):
    """
    CTC greedy decode.
    preds: torch.Tensor or np.ndarray with shape [B, T, C]
    Returns: list of (text, avg_conf)
    """
    preds = torch.nn.functional.softmax(preds, dim=2)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
    characters = [line.decode("utf-8").strip("\n").strip("\r\n") for line in lines]
    if use_space_char:
        characters.append(" ")
    characters = ["blank"] + characters

    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)

    results = []
    for b in range(preds_idx.shape[0]):
        char_list = []
        conf_list = []
        prev = None
        for t in range(preds_idx.shape[1]):
            idx = int(preds_idx[b, t])
            if idx == 0:
                prev = idx
                continue
            if prev is not None and idx == prev:
                continue
            char_list.append(characters[idx])
            conf_list.append(float(preds_prob[b, t]))
            prev = idx
        text = "".join(char_list)
        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        results.append((text, avg_conf))

    return results



if __name__ == "__main__":
    img_path = "./assets/group04_img10.png"
    weight_path = "./weights/ptocr_v5_mobile_rec.pth"
    dict_path = "./assets/ppocrv5_dict.txt"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    x = _preprocess_no_resize_bgr(img)
    x = torch.from_numpy(x).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPOCRv5MobileRec(character_dict_path=dict_path, use_space_char=True)
    model.load_weights(weight_path, map_location=device, strict=True)
    model.eval()
    model.to(device)
    x = x.to(device)

    with torch.no_grad():
        out = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(out.shape))
    results = decode_ctc(out, dict_path, use_space_char=True)
    for i, (text, conf) in enumerate(results):
        print(f"[{i}] text: {text}")
        print(f"[{i}] avg_conf: {conf:.4f}")
