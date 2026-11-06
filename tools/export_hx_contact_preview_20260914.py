import cv2

source = "/AI_DNS/guang.yang/TextSR/Lamu/N2N/evaluation/hx_v4_v5_r1_frozen_ocr_native_20260914/visuals/comparisons/001_HXL772_main_260730_row1_col1__lq_v4_v5.png"
image = cv2.imread(source)
height, width = image.shape[:2]
scale = min(1.0, 1000.0 / width)
preview = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
cv2.imwrite("/tmp/hx_v4_v5_contact_01.jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 68])
