import cv2
import numpy as np

def optical_flow_motion_blur(frame1, frame2, num_samples=10):
    """基于光流的运动模糊合成"""
    # 1. 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # 2. 创建累加器
    height, width = frame1.shape[:2]
    result = np.zeros_like(frame1, dtype=np.float32)

    # 3. 沿光流路径采样
    for i in range(num_samples):
        t = i / (num_samples - 1)  # 0到1之间
        # 计算当前偏移
        offset = flow * t

        # 创建映射矩阵
        map_x = np.tile(np.arange(width), (height, 1)).astype(np.float32)
        map_y = np.tile(np.arange(height), (width, 1)).T.astype(np.float32)

        map_x += offset[:, :, 0]
        map_y += offset[:, :, 1]

        # 重采样
        warped = cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 累加
        result += warped.astype(np.float32)

    # 4. 平均
    result = (result / num_samples).astype(np.uint8)
    return result


import numpy as np
import cv2

def nv21_to_yuv_planes(nv21_data, width, height):
    """将NV21数据分离为Y、U、V平面"""
    y_size = width * height
    uv_size = y_size // 4  # UV分量各占1/4

    # Y分量
    y_plane = nv21_data[:y_size].reshape(height, width)

    # UV交错分量 (VU交替存储，注意顺序：NV21是VU交替)
    uv_interleaved = nv21_data[y_size:y_size + 2*uv_size]

    # 分离U和V
    u_plane = np.zeros((height//2, width//2), dtype=np.uint8)
    v_plane = np.zeros((height//2, width//2), dtype=np.uint8)

    for i in range(uv_size):
        row = i // (width//2)
        col = i % (width//2)
        v_plane[row, col] = uv_interleaved[2*i]      # V分量
        u_plane[row, col] = uv_interleaved[2*i + 1]  # U分量

    return y_plane, u_plane, v_plane

def yuv_planes_to_nv21(y_plane, u_plane, v_plane):
    """将YUV平面合并为NV21格式"""
    height, width = y_plane.shape
    uv_height, uv_width = u_plane.shape

    # Y分量
    y_data = y_plane.flatten()

    # 交错UV分量 (V在前，U在后)
    uv_size = uv_height * uv_width
    uv_interleaved = np.zeros(2 * uv_size, dtype=np.uint8)

    for i in range(uv_size):
        row = i // uv_width
        col = i % uv_width
        uv_interleaved[2*i] = v_plane[row, col]      # V
        uv_interleaved[2*i + 1] = u_plane[row, col]  # U

    # 合并为NV21
    nv21_data = np.concatenate([y_data, uv_interleaved])
    return nv21_data

def motion_blur_nv21(nv21_frame1, nv21_frame2, width, height,
                     num_samples=5, blur_strength=1.0):
    """
    对NV21格式的两帧图像合成运动模糊

    参数：
        nv21_frame1: 第一帧NV21数据
        nv21_frame2: 第二帧NV21数据
        width, height: 图像尺寸
        num_samples: 沿光流路径的采样点数
        blur_strength: 模糊强度系数

    返回：
        合成后的NV21数据
    """
    # 1. 分离YUV分量
    y1, u1, v1 = nv21_to_yuv_planes(nv21_frame1, width, height)
    y2, u2, v2 = nv21_to_yuv_planes(nv21_frame2, width, height)

    # 2. 在Y分量上计算光流（亮度信息最丰富）
    flow = cv2.calcOpticalFlowFarneback(
        y1, y2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # 3. 调整模糊强度
    flow = flow * blur_strength

    # 4. 生成Y分量的运动模糊
    y_blurred = motion_blur_channel(y1, flow, num_samples)

    # 5. 处理UV分量（需要特殊处理，因为分辨率是Y的一半）
    u_blurred, v_blurred = motion_blur_uv_channels(u1, v1, flow, num_samples)

    # 6. 合并回NV21格式
    nv21_blurred = yuv_planes_to_nv21(y_blurred, u_blurred, v_blurred)

    return nv21_blurred

def motion_blur_channel(channel, flow, num_samples):
    """对单个通道（Y分量）进行运动模糊"""
    height, width = channel.shape
    result = np.zeros((height, width), dtype=np.float32)

    # 创建基础坐标网格
    x_base, y_base = np.meshgrid(np.arange(width), np.arange(height))

    for i in range(num_samples):
        # 采样位置（从0到1）
        t = i / max(num_samples - 1, 1)

        # 计算当前采样位置的坐标
        map_x = (x_base + flow[:, :, 0] * t).astype(np.float32)
        map_y = (y_base + flow[:, :, 1] * t).astype(np.float32)

        # 重采样
        sampled = cv2.remap(
            channel.astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        result += sampled

    # 平均并转换回uint8
    result /= num_samples
    return np.clip(result, 0, 255).astype(np.uint8)

def motion_blur_uv_channels(u_plane, v_plane, flow_y, num_samples):
    """处理UV分量的运动模糊（需要下采样光流）"""
    uv_height, uv_width = u_plane.shape

    # 1. 将光流下采样到UV分辨率
    flow_uv = cv2.resize(flow_y, (uv_width, uv_height), interpolation=cv2.INTER_AREA)
    flow_uv = flow_uv / 2.0  # 因为UV分辨率是Y的一半

    # 2. 分别处理U和V通道
    u_blurred = motion_blur_channel(u_plane, flow_uv, num_samples)
    v_blurred = motion_blur_channel(v_plane, flow_uv, num_samples)

    return u_blurred, v_blurred


def fast_motion_blur_nv21(nv21_frame1, nv21_frame2, width, height,
                          num_samples=3, use_gpu=False):
    """
    优化版本：更快的运动模糊合成

    参数：
        use_gpu: 是否使用CUDA加速（如果OpenCV编译了CUDA支持）
    """
    # 分离Y分量
    y_size = width * height
    y1 = nv21_frame1[:y_size].reshape(height, width)
    y2 = nv21_frame2[:y_size].reshape(height, width)

    # 计算光流
    if use_gpu and hasattr(cv2, 'cuda'):
        # GPU版本（如果可用）
        gpu_y1 = cv2.cuda_GpuMat(y1)
        gpu_y2 = cv2.cuda_GpuMat(y2)

        # 创建光流对象
        farneback = cv2.cuda_FarnebackOpticalFlow.create(
            5, 0.5, False, 15, 3, 5, 1.2, 0
        )
        gpu_flow = farneback.calc(gpu_y1, gpu_y2, None)
        flow = gpu_flow.download()
    else:
        # CPU版本
        flow = cv2.calcOpticalFlowFarneback(
            y1, y2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

    # 快速运动模糊（简化版本，只处理Y分量，UV使用第一帧）
    y_blurred = fast_channel_blur(y1, flow, num_samples)

    # UV分量直接使用第一帧（对于实时应用，这是个合理的折中）
    uv_data = nv21_frame1[y_size:]

    # 合并结果
    nv21_blurred = np.concatenate([y_blurred.flatten(), uv_data])

    return nv21_blurred

def fast_channel_blur(channel, flow, num_samples):
    """快速通道模糊，使用更高效的实现"""
    height, width = channel.shape

    if num_samples == 1:
        return channel

    # 使用整数采样位置，避免重复计算
    result = np.zeros_like(channel, dtype=np.float32)

    # 只采样几个关键位置（0, 0.5, 1.0）
    sample_weights = [0.25, 0.5, 0.25]  # 可以调整权重
    sample_positions = [0.0, 0.5, 1.0]

    for weight, t in zip(sample_weights, sample_positions):
        # 计算映射
        map_x = np.arange(width).astype(np.float32)
        map_y = np.arange(height).astype(np.float32)
        map_x, map_y = np.meshgrid(map_x, map_y)

        map_x += flow[:, :, 0] * t
        map_y += flow[:, :, 1] * t

        # 重采样
        sampled = cv2.remap(
            channel, map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        result += sampled * weight

    return np.clip(result, 0, 255).astype(np.uint8)

def example_usage():
    """使用示例：从文件读取NV21数据并生成运动模糊"""
    import os

    # 假设有两个NV21格式的文件
    width, height = 640, 480
    y_size = width * height
    uv_size = y_size // 2
    total_size = y_size + uv_size

    # 从文件读取NV21数据
    with open('frame1.nv21', 'rb') as f:
        nv21_frame1 = np.frombuffer(f.read(total_size), dtype=np.uint8)

    with open('frame2.nv21', 'rb') as f:
        nv21_frame2 = np.frombuffer(f.read(total_size), dtype=np.uint8)

    # 方法1：完整处理
    print("完整处理YUV通道...")
    nv21_blurred = motion_blur_nv21(
        nv21_frame1, nv21_frame2,
        width, height,
        num_samples=5,
        blur_strength=1.0
    )

    # 保存结果
    with open('blurred.nv21', 'wb') as f:
        f.write(nv21_blurred.tobytes())

    # 方法2：快速处理（只处理Y分量）
    print("快速处理（只Y通道）...")
    nv21_blurred_fast = fast_motion_blur_nv21(
        nv21_frame1, nv21_frame2,
        width, height,
        num_samples=3
    )

    # 可视化（转换为RGB查看）
    def nv21_to_bgr(nv21_data, width, height):
        """NV21转BGR用于显示"""
        bgr = cv2.cvtColor(
            nv21_data.reshape(int(height * 1.5), width),
            cv2.COLOR_YUV2BGR_NV21
        )
        return bgr

    # 显示结果
    original_bgr = nv21_to_bgr(nv21_frame1, width, height)
    blurred_bgr = nv21_to_bgr(nv21_blurred, width, height)

    cv2.imshow('Original', original_bgr)
    cv2.imshow('Motion Blurred', blurred_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return nv21_blurred

def simulate_nv21_from_rgb(rgb_image):
    """从RGB图像模拟生成NV21数据（用于测试）"""
    # RGB转YUV
    yuv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV_I420)
    height, width = rgb_image.shape[:2]

    # I420转NV21
    nv21 = np.zeros(int(width * height * 1.5), dtype=np.uint8)

    # Y分量
    y_size = width * height
    nv21[:y_size] = yuv[:y_size]

    # UV分量（I420是U平面在前，V平面在后；NV21是VU交错）
    uv_start = y_size
    uv_size = y_size // 4

    # I420: [U平面, V平面]
    u_plane = yuv[y_size:y_size + uv_size]
    v_plane = yuv[y_size + uv_size:]

    # NV21: VU交错
    for i in range(uv_size):
        nv21[uv_start + 2*i] = v_plane[i]     # V
        nv21[uv_start + 2*i + 1] = u_plane[i] # U

    return nv21, width, height

def test_with_simulated_data():
    """使用模拟数据测试"""
    # 创建两个测试图像
    img1 = cv2.imread(r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data\sr_20251121181205_input_00_rgb.png')
    img2 = cv2.imread(r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data\sr_20251121181205_input_05_rgb.png')

    # 模拟NV21数据
    nv21_1, width, height = simulate_nv21_from_rgb(img1)
    nv21_2, _, _ = simulate_nv21_from_rgb(img2)

    # 生成运动模糊
    nv21_blurred = motion_blur_nv21(
        nv21_1, nv21_2, width, height,
        num_samples=7, blur_strength=1.2
    )

    # 转换回RGB显示
    blurred_rgb = cv2.cvtColor(
        nv21_blurred.reshape(int(height * 1.5), width),
        cv2.COLOR_YUV2RGB_NV21
    )

    return img1, blurred_rgb



if __name__=="__main__":

    # # 读取两帧图像
    # frame1 = cv2.imread(r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data\sr_20251121181205_input_00_rgb.png')
    # frame2 = cv2.imread(r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data\sr_20251121181205_input_05_rgb.png')

    # # 生成运动模糊图像
    # blurred_image = optical_flow_motion_blur(frame1, frame2, num_samples=15)

    # # 保存结果
    # cv2.imwrite(r'E:\workspace\AISR\Real-ESRGAN\datasets\debug_data\0-5_motion_blurred.png', blurred_image)

    print("测试运动模糊合成...")
    original, blurred = test_with_simulated_data()

    cv2.imshow('Original (simulated)', original)
    cv2.imshow('Motion Blurred', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
