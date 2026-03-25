import os
import shutil


def process_depth2l():
    # 基础路径
    base_output = "/home/ljc/ntire/3D-UIR/output"

    # 七个场景文件夹（根据图片显示，建议使用小写）
    scenes = ['tsubaki', 'shirohana', 'natsume', 'midori', 'koharu', 'hinoki', 'futaba']

    for scene in scenes:
        # 1. 定义原始 depth2l 路径
        src_depth2l = os.path.join(base_output, scene, "test", "ours_30000", "depth2l")

        # 2. 定义目标位置路径 (往前移一个路径，到 test 下)
        dst_depth2l = os.path.join(base_output, scene, "test", "depth2l")

        if not os.path.exists(src_depth2l):
            print(f"跳过: 找不到路径 {src_depth2l}")
            continue

        print(f"正在处理场景: {scene}...")

        # 3. 对 depth2l 下的文件进行重命名并按顺序编号
        # 获取所有 .png 文件并进行排序，确保 0024 在 0025 前面
        files = sorted([f for f in os.listdir(src_depth2l) if f.endswith('.png')])

        for index, old_filename in enumerate(files, start=1):
            # 构造新名字：例如 tsubaki_0001.png
            new_filename = f"{scene}_{index:04d}.png"

            old_file_path = os.path.join(src_depth2l, old_filename)
            new_file_path = os.path.join(src_depth2l, new_filename)

            # 执行重命名
            os.rename(old_file_path, new_file_path)

        # 4. 将整个文件夹往前移动一个位置
        # 如果目标位置已存在同名文件夹，先删掉旧的以防止嵌套
        if os.path.exists(dst_depth2l):
            shutil.rmtree(dst_depth2l)

        shutil.move(src_depth2l, dst_depth2l)
        print(f"完成: {scene} 的 depth2l 已移动至 {dst_depth2l}")


if __name__ == "__main__":
    process_depth2l()