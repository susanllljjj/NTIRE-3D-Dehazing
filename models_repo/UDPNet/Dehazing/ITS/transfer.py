# import os
# from PIL import Image
# from tqdm import tqdm
#
#
# def convert_images():
#     # 目标目录
#     target_dir = "/data2/gy/data/ljc/Akikaze/test/gt"
#
#     if not os.path.exists(target_dir):
#         print(f"错误: 找不到目录 {target_dir}")
#         return
#
#     # 获取所有 jpg/JPG 文件
#     files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
#
#     if not files:
#         print("没有发现需要转换的 JPG 文件。")
#         return
#
#     print(f"开始转换 {len(files)} 张图片...")
#
#     for filename in tqdm(files):
#         # 构造完整路径
#         old_path = os.path.join(target_dir, filename)
#
#         # 构造新文件名 (替换后缀名为 .png)
#         new_filename = os.path.splitext(filename)[0] + ".png"
#         new_path = os.path.join(target_dir, new_filename)
#
#         try:
#             # 打开并转换
#             with Image.open(old_path) as img:
#                 img.save(new_path, "PNG")
#
#             # 转换成功后删除原 JPG 文件
#             os.remove(old_path)
#         except Exception as e:
#             print(f"处理文件 {filename} 时出错: {e}")
#
#     print("转换完成！")
#
#
# if __name__ == "__main__":
#     convert_images()



import os
from PIL import Image
from tqdm import tqdm

# 定义需要转换的根目录
base_path = "/data2/gy/data/ljc/Akikaze/split_data"
splits = ['train', 'test']  # 如果你还有 test 文件夹，也加上 'test'

for split in splits:
    gt_path = os.path.join(base_path, split, 'gt')
    if not os.path.exists(gt_path):
        continue

    print(f"正在转换 {split}/gt 中的图片...")
    files = os.listdir(gt_path)

    for filename in tqdm(files):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(gt_path, filename)

            # 打开图片并保存为 png
            img = Image.open(img_path)
            new_filename = os.path.splitext(filename)[0] + ".png"
            new_path = os.path.join(gt_path, new_filename)

            img.save(new_path)

            # 删除旧的 jpg 文件
            os.remove(img_path)

print("转换完成！现在所有 GT 文件都是 .png 格式了。")