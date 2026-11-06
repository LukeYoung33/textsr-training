def get_ocr_loss_scale(current_iter, start_iter=0, ramp_iters=0):
    """Return the external OCR curriculum scale without mutating the loss module."""
    current_iter = int(current_iter)
    start_iter = int(start_iter)
    ramp_iters = int(ramp_iters)
    if current_iter < start_iter:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return min(max((current_iter - start_iter) / float(ramp_iters), 0.0), 1.0)
