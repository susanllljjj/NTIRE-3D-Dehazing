#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, depth_with_colormap
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
import numpy as np
# 1. 确保能导入 main_system
sys.path.append("/home/ljc/ntire")
from main_system import NTIREFinalSystem
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    # 1. 核心步骤：对相机视角按原始图片名称的数字大小进行排序
    # 这样能保证 idx=0 对应 0001.png, idx=1 对应 0002.png...
    def get_image_num(view):
        try:
            # 提取文件名里的数字，例如从 "0023.png" 或 "0023" 提取出 23
            name_only = os.path.basename(view.image_name).split('.')[0]
            return int(name_only)
        except:
            return view.image_name  # 如果不是数字则按字符串排

    views.sort(key=get_image_num)

    # 2. 获取前缀 (例如 "tsubaki_")
    prefix = os.path.basename(model_path) + "_"

    # 3. 重新定义路径 (去掉 ours_{iteration})
    render_path = os.path.join(model_path, name, "hazy")
    gts_path = os.path.join(model_path, name, "gt")
    clear_path = os.path.join(model_path, name, "clear")
    depth_path = os.path.join(model_path, name, "depth")
    attenuation_path = os.path.join(model_path, name, "attenuation")
    backscatter_path = os.path.join(model_path, name, "backscatter")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(clear_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(attenuation_path, exist_ok=True)
    makedirs(backscatter_path, exist_ok=True)

    render_time_list = []
    fps_list = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # 4. 强制按排序后的顺序重命名 (tsubaki_0001.png, tsubaki_0002.png...)
        final_image_name = "{}{:04d}.png".format(prefix, idx + 1)

        torch.cuda.synchronize()
        start = time.time()

        render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp,
                            separate_sh=separate_sh)

        torch.cuda.synchronize()
        end = time.time()

        render_time_ms = (end - start) * 1000
        render_time_list.append(render_time_ms)
        fps_list.append(1000 / render_time_ms)

        rendering, clear, depth, attenuation, backscatter = render_pkg["render"], render_pkg["restore_scene"], \
            render_pkg["depth"], render_pkg["attenuation"], render_pkg["backscatter"]

        depth = depth_with_colormap(depth, colormap='turbo')
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
            clear = clear[..., clear.shape[-1] // 2:]
            depth = depth[..., depth.shape[-1] // 2:]

        # 保存图片
        torchvision.utils.save_image(rendering, os.path.join(render_path, final_image_name))
        torchvision.utils.save_image(gt, os.path.join(gts_path, final_image_name))
        torchvision.utils.save_image(clear, os.path.join(clear_path, final_image_name))
        torchvision.utils.save_image(depth, os.path.join(depth_path, final_image_name))
        torchvision.utils.save_image(attenuation, os.path.join(attenuation_path, final_image_name))
        torchvision.utils.save_image(backscatter, os.path.join(backscatter_path, final_image_name))

    # 渲染循环结束后
    if name == "test":  # 只有在跑测试集渲染时才触发
        #print("\n[3D-UIR] 渲染任务结束，正在移交去雾系统...")

        # 2. 获取 model_path 的绝对路径
        abs_model_path = os.path.abspath(model_path)

        # 3. 启动去雾流程
        try:
            dehaze_app = NTIREFinalSystem()
            dehaze_app.run_all(abs_model_path)
        except Exception as e:
            print(f"❌ 去雾流程触发失败: {e}")



# def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     clear_path = os.path.join(model_path, name, "ours_{}".format(iteration), "clear")
#     depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
#     attenuation_path = os.path.join(model_path, name, "ours_{}".format(iteration), "attenuation")
#     backscatter_path = os.path.join(model_path, name, "ours_{}".format(iteration), "backscatter")
#
#     makedirs(render_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
#     makedirs(clear_path, exist_ok=True)
#     makedirs(depth_path, exist_ok=True)
#     makedirs(attenuation_path, exist_ok=True)
#     makedirs(backscatter_path, exist_ok=True)
#
#     render_time_list = []
#     fps_list = []
#
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#
#         # 如果名字里没点（没后缀），就强行加个 .png
#         if "." not in view.image_name:
#             view.image_name = view.image_name + ".png"
#
#         torch.cuda.synchronize()
#         start = time.time()
#
#         render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
#
#         torch.cuda.synchronize()
#         end = time.time()
#
#         render_time_ms = (end - start) * 1000
#         render_time_list.append(render_time_ms)
#
#         fps = 1000 / render_time_ms
#         fps_list.append(fps)
#
#         rendering, clear, depth, attenuation, backscatter = render_pkg["render"], render_pkg["restore_scene"], \
#             render_pkg["depth"], render_pkg["attenuation"], render_pkg["backscatter"]
#
#         depth = depth_with_colormap(depth, colormap='turbo')  # turbo as NeRFStudio
#         gt = view.original_image[0:3, :, :]
#
#         if args.train_test_exp:
#             rendering = rendering[..., rendering.shape[-1] // 2:]
#             gt = gt[..., gt.shape[-1] // 2:]
#             clear = clear[..., clear.shape[-1] // 2:]
#             depth = depth[..., depth.shape[-1] // 2:]
#
#         mean_fps = np.mean(fps_list[5:])
#
#         torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name))
#         # # 检查文件名里是否有点，如果没有则手动加一个 .png
#         # save_path = os.path.join(render_path, view.image_name)
#         # if not save_path.endswith(".png") and not save_path.endswith(".jpg"):
#         #     save_path += ".png"
#         #torchvision.utils.save_image(rendering, save_path)
#         torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name))
#         torchvision.utils.save_image(clear, os.path.join(clear_path, view.image_name))
#         torchvision.utils.save_image(depth, os.path.join(depth_path, view.image_name))
#         torchvision.utils.save_image(attenuation, os.path.join(attenuation_path, view.image_name))
#         torchvision.utils.save_image(backscatter, os.path.join(backscatter_path, view.image_name))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), True, args.skip_test, SPARSE_ADAM_AVAILABLE)