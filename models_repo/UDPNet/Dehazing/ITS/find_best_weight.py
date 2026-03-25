import cv2
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 1. 路径设置
path_udp = "/home/ljc/UDPNet/Dehazing/ITS/results/Akikaze_Final_Evaluation/test"
path_dehaze = "/home/ljc/DehazeFormer/results/Akikaze/smoke_finetune_v1/dehazeformer-b/imgs"
path_gt = "/data_nvme/ljc/smoke_validation/validation/Akikaze/GT"

# 2. 获取文件列表（以 GT 文件夹为基准）
files = [f for f in os.listdir(path_gt) if f.lower().endswith('.jpg')]
print(f"共检测到 {len(files)} 张图片进行评估。")


# ... 前面的代码保持不变 ...

def evaluate_ensemble(w):
    total_psnr = 0
    total_ssim = 0
    valid_count = 0

    for idx, filename in enumerate(files):
        f_gt = os.path.join(path_gt, filename)
        f_udp = os.path.join(path_udp, filename)
        f_dehaze = os.path.join(path_dehaze, filename)

        if os.path.exists(f_udp) and os.path.exists(f_dehaze):
            img_gt = cv2.imread(f_gt)
            img_udp = cv2.imread(f_udp)
            img_dehaze = cv2.imread(f_dehaze)

            # --- 加速优化 1: 如果图片太大，缩小尺寸计算权重 (可选) ---
            # 如果你只想快速看哪个权重好，取消下面三行的注释
            # img_gt = cv2.resize(img_gt, (512, 512))
            # img_udp = cv2.resize(img_udp, (512, 512))
            # img_dehaze = cv2.resize(img_dehaze, (512, 512))

            img_ens = cv2.addWeighted(img_udp.astype(np.float32), w,
                                      img_dehaze.astype(np.float32), (1 - w), 0)
            img_ens = np.clip(img_ens, 0, 255).astype(np.uint8)

            # PSNR 计算很快
            cur_psnr = psnr(img_gt, img_ens, data_range=255)

            # --- 加速优化 2: SSIM 极其耗时，先只打印 PSNR 看看 ---
            # 如果你真的很急，可以先把下面这行 SSIM 注释掉，设置 cur_ssim = 0
            cur_ssim = ssim(img_gt, img_ens, data_range=255, channel_axis=-1)

            total_psnr += cur_psnr
            total_ssim += cur_ssim
            valid_count += 1

            # 打印当前是第几张图，让你心里有数
            print(f"[{idx + 1}/25]", end=" ", flush=True)

    return total_psnr / valid_count, total_ssim / valid_count





# 3. 遍历权重搜索 (步长 0.05)
best_psnr = 0
best_ssim = 0
best_w_psnr = 0
weights = np.linspace(0, 1, 21)  # 0, 0.05, 0.1, ..., 1.0

# ... 修改主循环打印方式 ...
print(f"{'Weight (UDP)':<15} | {'PSNR':<10} | {'SSIM':<10}")
print("-" * 45)

for w in weights:
    print(f"Testing w={w:.2f}: ", end="", flush=True)  # 先打印当前在测哪个权重
    avg_psnr, avg_ssim = evaluate_ensemble(w)
    print(f"\r{w:15.2f} | {avg_psnr:10.4f} | {avg_ssim:10.4f}")  # 覆盖掉进度点，显示正式结果

# ... 后面代码保持不变 ...

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        best_ssim = avg_ssim
        best_w_psnr = w

print("-" * 45)
print(f"最优权重 (基于 PSNR): {best_w_psnr:.2f}")
print(f"最高 PSNR: {best_psnr:.4f}, 对应 SSIM: {best_ssim:.4f}")