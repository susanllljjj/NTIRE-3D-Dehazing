import os
import random
import shutil

# 1. 配置
source_base = '/data_nvme/ljc/smoke_average'
target_base = '/data_nvme/ljc/smoke_average_split'

# 创建结构
for split in ['train', 'test']:
    for sub in ['hazy', 'gt', 'depth2l']:
        os.makedirs(os.path.join(target_base, split, sub), exist_ok=True)

# 2. 获取所有 hazy 文件列表
hazy_dir = os.path.join(source_base, 'hazy')
all_files = [f for f in os.listdir(hazy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.seed(42)
random.shuffle(all_files)

# 3. 划分 (90% 训练, 10% 验证)
# 因为你已经是微调阶段且数据较多，验证集不用留太多
split_idx = int(len(all_files) * 0.9)
train_files = all_files[:split_idx]
test_files = all_files[split_idx:]


def distribute(file_list, split_name):
    print(f"正在拷贝 {split_name} 数据...")
    for f in file_list:
        name_no_ext = os.path.splitext(f)[0]

        # 拷贝 hazy
        shutil.copy(os.path.join(source_base, 'hazy', f),
                    os.path.join(target_base, split_name, 'hazy', f))

        # 拷贝 GT (注意这里从 source_base/GT 读，拷到 target_base/split/gt)
        # 确保后缀匹配，如果 GT 也是 JPG 或是 PNG
        gt_f = f  # 假设名字完全一样
        shutil.copy(os.path.join(source_base, 'GT', gt_f),
                    os.path.join(target_base, split_name, 'gt', gt_f))

        # 拷贝 depth2l (通常是 .png)
        depth_f = name_no_ext + '.png'
        if os.path.exists(os.path.join(source_base, 'depth2l', depth_f)):
            shutil.copy(os.path.join(source_base, 'depth2l', depth_f),
                        os.path.join(target_base, split_name, 'depth2l', depth_f))


distribute(train_files, 'train')
distribute(test_files, 'test')
print("划分完成！")