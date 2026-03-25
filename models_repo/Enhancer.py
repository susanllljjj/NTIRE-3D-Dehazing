# /home/ljc/ntire/models_repo/Enhancer.py
import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import argparse


class SceneRefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine_conv = nn.Module()
        self.refine_conv.register_parameter('weight', nn.Parameter(torch.ones(3, 3, 1, 1)))
        self.refine_conv.register_parameter('bias', nn.Parameter(torch.ones(3, 1, 1)))

    def forward(self, x):
        return x + self.refine_conv.bias


def run_refine():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    device = torch.device('cuda')
    all_data = torch.load(args.weights, map_location=device)
    refine_weights = all_data['weights_refine']

    os.makedirs(args.output, exist_ok=True)
    model = SceneRefineNet().to(device)

    for img_name in sorted(os.listdir(args.input)):
        prefix = img_name.split('_')[0]
        if prefix in ['futaba', 'shirohana', 'tsubaki']:
            img = cv2.imread(os.path.join(args.input, img_name)).astype(np.float32) / 255.0
            img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            w = refine_weights[prefix][img_name]
            model.refine_conv.bias = torch.nn.Parameter(torch.zeros(w['refine_conv.bias'].shape).to(device))
            model.load_state_dict(w)

            with torch.no_grad():
                out = model(img_t)

            # 保存
            res = (np.clip(out.detach().squeeze().permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, img_name), res)
            print(f"RefineNet 已增强: {img_name}")


if __name__ == "__main__":
    run_refine()