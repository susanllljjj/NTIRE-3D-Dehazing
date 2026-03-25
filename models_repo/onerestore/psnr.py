import os
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path):
    """
    读取图像并转成 numpy 数组
    """
    img = Image.open(path)
    return np.array(img)


def calc_psnr(img1, img2):
    """
    计算两张图的 PSNR
    """
    if img1.shape != img2.shape:
        raise ValueError(f"shape 不一致: {img1.shape} vs {img2.shape}")

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    # 默认按 8-bit 图像处理
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def collect_files(root):
    """
    收集文件夹下所有图像文件，返回:
    {
        相对路径: 绝对路径
    }
    """
    root = Path(root)
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            rel_path = path.relative_to(root).as_posix()
            files[rel_path] = path
    return files


def main():
    parser = argparse.ArgumentParser(description="计算两个文件夹中对应图像文件的 PSNR")
    parser.add_argument("dir1", type=str, help="文件夹1")
    parser.add_argument("dir2", type=str, help="文件夹2")
    parser.add_argument("--save", type=str, default=None, help="保存结果到 txt 文件")
    args = parser.parse_args()

    files1 = collect_files(args.dir1)
    files2 = collect_files(args.dir2)

    common_keys = sorted(set(files1.keys()) & set(files2.keys()))
    only_in_1 = sorted(set(files1.keys()) - set(files2.keys()))
    only_in_2 = sorted(set(files2.keys()) - set(files1.keys()))

    lines = []
    psnr_list = []

    lines.append(f"folder1: {args.dir1}")
    lines.append(f"folder2: {args.dir2}")
    lines.append(f"匹配文件数: {len(common_keys)}")
    lines.append("")

    if only_in_1:
        lines.append("仅在 folder1 中存在:")
        for k in only_in_1:
            lines.append(f"  {k}")
        lines.append("")

    if only_in_2:
        lines.append("仅在 folder2 中存在:")
        for k in only_in_2:
            lines.append(f"  {k}")
        lines.append("")

    for rel_path in common_keys:
        p1 = files1[rel_path]
        p2 = files2[rel_path]

        try:
            img1 = load_image(p1)
            img2 = load_image(p2)
            psnr = calc_psnr(img1, img2)
            psnr_list.append(psnr)
            psnr_str = "inf" if math.isinf(psnr) else f"{psnr:.4f}"
            line = f"{rel_path}: PSNR = {psnr_str} dB"
        except Exception as e:
            line = f"{rel_path}: 计算失败 ({e})"

        print(line)
        lines.append(line)

    lines.append("")
    if psnr_list:
        finite_vals = [x for x in psnr_list if not math.isinf(x)]
        if finite_vals:
            avg_psnr = sum(finite_vals) / len(finite_vals)
            lines.append(f"平均 PSNR: {avg_psnr:.4f} dB")
            print(f"\n平均 PSNR: {avg_psnr:.4f} dB")
        else:
            lines.append("平均 PSNR: inf")
            print("\n平均 PSNR: inf")
    else:
        lines.append("没有成功计算的匹配文件。")
        print("\n没有成功计算的匹配文件。")

    if args.save is not None:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"结果已保存到: {args.save}")


if __name__ == "__main__":
    main()