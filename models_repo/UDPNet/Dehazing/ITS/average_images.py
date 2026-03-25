import os
import numpy as np
from PIL import Image

# 路径设置
dir_jpg = "/data_nvme/ljc/dehazeformer_results/Midori/smoke_finetune_v1/dehazeformer-b/imgs"
dir_png = "/home/ljc/UDPNet/Dehazing/ITS/results/Midori_Result/test"
dst_dir = "/data_nvme/ljc/smoke_average/GT"
prefix = "Midori_"

# 创建目标目录
os.makedirs(dst_dir, exist_ok=True)

# 获取 dir_jpg 中的文件名列表 (作为基准)
files_jpg = [f for f in os.listdir(dir_jpg) if f.lower().endswith('.jpg')]

print(f"开始处理，预计处理 {len(files_jpg)} 张图片...")

count = 0
for filename in files_jpg:
    # 获取不带后缀的文件名
    stem = os.path.splitext(filename)[0]

    # 构建对应的 png 路径
    path_jpg = os.path.join(dir_jpg, filename)
    path_png = os.path.join(dir_png, stem + ".png")

    # 检查 png 文件是否存在
    if os.path.exists(path_png):
        try:
            # 读取图片并转为 RGB 模式
            img_jpg = Image.open(path_jpg).convert('RGB')
            img_png = Image.open(path_png).convert('RGB')

            # 如果尺寸不一致，以 JPG 为准进行缩放（通常数据集尺寸应该是一致的）
            if img_jpg.size != img_png.size:
                img_png = img_png.resize(img_jpg.size, Image.LANCZOS)

            # 转换为 numpy 数组进行计算
            arr_jpg = np.array(img_jpg, dtype=np.float32)
            arr_png = np.array(img_png, dtype=np.float32)

            # 求平均值
            avg_arr = (arr_jpg + arr_png) / 2.0

            # 转回图像并保存
            avg_img = Image.fromarray(avg_arr.astype(np.uint8))

            save_name = f"{prefix}{stem}.JPG"
            avg_img.save(os.path.join(dst_dir, save_name), quality=95, subsampling=0)

            count += 1
            if count % 100 == 0:
                print(f"已完成 {count} 张...")

        except Exception as e:
            print(f"处理 {stem} 时出错: {e}")
    else:
        print(f"跳过: 未找到对应的 PNG 文件 {path_png}")

print(f"全部完成！成功生成 {count} 张平均图像，存放在: {dst_dir}")