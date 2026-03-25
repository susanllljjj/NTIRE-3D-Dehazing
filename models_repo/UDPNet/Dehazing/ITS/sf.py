import os
from PIL import Image

# 你的数据根目录
base_path = '/data_nvme/ljc/smoke_validation_udp/Akikaze/test'
# 想要缩放的目标尺寸
target_size = (512, 512)

folders = ['hazy', 'gt', 'depth2l']

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        continue

    # 创建一个存放缩放后图片的临时目录
    output_path = os.path.join(base_path, folder + '_512')
    os.makedirs(output_path, exist_ok=True)

    print(f"正在处理文件夹: {folder}...")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            # 使用 LANCZOS 滤镜保证缩放质量
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(os.path.join(output_path, filename))

print("全部缩放完成！缩放后的图片在 *_512 文件夹中。")