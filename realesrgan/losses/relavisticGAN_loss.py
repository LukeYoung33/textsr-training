import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativisticGANLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ganloss_mask=1.0):
        super(RelativisticGANLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ganloss_mask = ganloss_mask

    def forward(self, fake_logits, real_logits, is_disc=False):
        if is_disc:
            relativistic_logits = real_logits - fake_logits
            loss = (F.softplus(-relativistic_logits) * self.ganloss_mask).mean()
        else:
            relativistic_logits = fake_logits - real_logits
            loss = self.loss_weight * (F.softplus(-relativistic_logits) * self.ganloss_mask).mean()
        return loss

if __name__=="__main__":
    import cv2
    gt = r"E:\workspace\AISR\test_1021\show_cases\ip11pro_output_TextSR_renew_40.JPG"
    pred_lr = r"E:\workspace\AISR\test_1021\show_cases\ip11pro_output_TextSR_renew_40_degraded_1.75_21_16_0.492_lr_up.png"
    pred_ocr = r"E:\workspace\AISR\test_1021\show_cases\ip11pro_output_TextSR_renew_40_degraded_1.75_21_16_0.779_OCR.png"

    # target = (torch.from_numpy(cv2.imread(gt)) / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    target = (torch.from_numpy(cv2.imread(pred_lr)) / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    pred_ocr = (torch.from_numpy(cv2.imread(pred_ocr)) / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

    cri = RelativisticGANLoss()
    # lr_loss = cri(pred_lr, target, is_disc=False)
    # print(lr_loss)
    ocr_loss = cri(pred_ocr, target, is_disc=False)
    print(ocr_loss)
