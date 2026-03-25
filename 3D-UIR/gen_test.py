import json
import os
from PIL import Image

# 1. 设定路径
base_path = "/data_nvme/ljc/IHDCP/Akikaze"
train_img_dir = os.path.join(base_path, "train")
test_img_dir = os.path.join(base_path, "test")
test_depth_dir = os.path.join(base_path, "depths")
json_path = os.path.join(base_path, "transforms_test.json")

os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_depth_dir, exist_ok=True)

# 2. 获取原图（训练集）的尺寸
# 随便找一张训练集的图
sample_img_name = os.listdir(train_img_dir)[0]
sample_img = Image.open(os.path.join(train_img_dir, sample_img_name))
W, H = sample_img.size
print(f"检测到原图尺寸为: {W} x {H}")

# 3. 创建对应的全黑占位图
dummy_img = Image.new('RGB', (W, H), (0, 0, 0))
dummy_depth = Image.new('L', (W, H), (0))

# 4. 读取 JSON 并生成
with open(json_path, 'r') as f:
    data = json.load(f)

print(f"开始为 {len(data['frames'])} 个测试视角生成占位图...")
for frame in data['frames']:
    # 提取文件名
    file_base = os.path.basename(frame['file_path'])

    # 保存占位原图python
    save_img_path = os.path.join(test_img_dir, f"{file_base}.png")
    dummy_img.save(save_img_path)

    # 保存占位深度图 (3D-UIR 训练需要对应的深度占位图)
    save_depth_path = os.path.join(test_depth_dir, f"{file_base}.png")
    dummy_depth.save(save_depth_path)

print("✅ 所有占位文件已生成，尺寸完全对齐！")
print("✅ 所有占位文件已生成，尺寸完全对齐！")