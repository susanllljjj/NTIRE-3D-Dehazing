import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 路径设置 ---
# 注意：请确保路径最后没有多余空格
gt_dir = '/data_nvme/ljc/smoke_validation/validation/Akikaze/GT'
res_dir = '/data_nvme/ljc/IPC_results/split_data/Akikaze'

# 获取结果目录下所有的 .png 文件
res_files = [f for f in os.listdir(res_dir) if f.lower().endswith('.png')]
res_files.sort()

psnr_list = []
ssim_list = []
not_found_count = 0

print(f"开始计算，共找到 {len(res_files)} 张结果图...")

for file_png in tqdm(res_files):
    # 1. 获取不带后缀的文件名 (例如从 '001.png' 得到 '001')
    base_name = os.path.splitext(file_png)[0]

    # 2. 构造对应的 GT 文件名 (这里匹配 .JPG)
    # 注意：如果你的 GT 后缀是小写 .jpg，请修改下面
    file_gt = base_name + ".JPG"

    res_path = os.path.join(res_dir, file_png)
    gt_path = os.path.join(gt_dir, file_gt)

    # 3. 检查 GT 文件是否存在 (处理大小写敏感问题)
    if not os.path.exists(gt_path):
        # 尝试小写后缀名作为备选
        gt_path = os.path.join(gt_dir, base_name + ".jpg")
        if not os.path.exists(gt_path):
            not_found_count += 1
            continue

    # 4. 读取图片
    img_res = cv2.imread(res_path)
    img_gt = cv2.imread(gt_path)

    if img_res is None or img_gt is None:
        print(f"警告: 无法读取图片 {base_name}")
        continue

    # 5. 统一尺寸 (Resizere 到 GT 的大小，确保对齐)
    if img_res.shape != img_gt.shape:
        img_res = cv2.resize(img_res, (img_gt.shape[1], img_gt.shape[0]))

    # 6. 计算指标 (RGB 空间)
    # skimage 的 ssim 在 0.19+ 版本中建议使用 channel_axis=2
    cur_psnr = psnr(img_gt, img_res, data_range=255)
    cur_ssim = ssim(img_gt, img_res, data_range=255, channel_axis=2)

    psnr_list.append(cur_psnr)
    ssim_list.append(cur_ssim)

# 7. 输出最终平均值
if len(psnr_list) > 0:
    print("\n" + "=" * 40)
    print(f"结果目录: {res_dir}")
    print(f"GT 目录:  {gt_dir}")
    print(f"成功处理: {len(psnr_list)} 张")
    print(f"未匹配到: {not_found_count} 张")
    print("-" * 40)
    print(f"平均 PSNR: {np.mean(psnr_list):.4f}")
    print(f"平均 SSIM: {np.mean(ssim_list):.4f}")
    print("=" * 40)
else:
    print("错误：没有成功处理任何图片，请检查文件名匹配规则。")