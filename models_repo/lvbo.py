# /home/ljc/ntire/models_repo/lvbo.py
import os
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入文件夹")
    parser.add_argument("--output", type=str, required=True, help="输出文件夹")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    kernel_size = (5, 5) # 高斯核
    sigma_x = 0

    for filename in os.listdir(input_dir):
        # 排除文件夹，只处理图片
        if filename.lower().endswith(image_exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            img = cv2.imread(input_path)
            if img is None: continue

            # 执行高斯滤波
            smoothed = cv2.GaussianBlur(img, kernel_size, sigma_x)
            cv2.imwrite(output_path, smoothed)
            print(f" 已完成平滑滤波: {filename}")

if __name__ == "__main__":
    main()

# import os
# import cv2
#
# # 输入和输出文件夹
# input_dir = "images"
# output_dir = "output_images"
#
# # 如果输出文件夹不存在，就创建
# os.makedirs(output_dir, exist_ok=True)
#
# # 支持的图片格式
# image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
#
# # 高斯核大小，必须是奇数，比如 (3,3), (5,5), (7,7)
# kernel_size = (5, 5)
#
# # sigmaX=0 表示由 OpenCV 根据核大小自动计算
# sigma_x = 0
#
# for filename in os.listdir(input_dir):
#     if filename.lower().endswith(image_exts):
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#
#         # 读取图像
#         img = cv2.imread(input_path)
#         if img is None:
#             print(f"无法读取: {input_path}")
#             continue
#
#         # 高斯滤波，输出图像尺寸与输入保持一致
#         smoothed = cv2.GaussianBlur(img, kernel_size, sigma_x)
#
#         # 保存结果
#         cv2.imwrite(output_path, smoothed)
#         print(f"已处理: {filename}")
#
# print("全部处理完成。")