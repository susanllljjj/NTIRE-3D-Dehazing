import os
import time
import torch
import argparse
import sys
from utils import Adder
from pytorch_msssim import ssim
import torch.nn.functional as f
from data import test_dataloader
from torch.backends import cudnn
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
#from models.FSNet import build_net
from models.FSNet_UDPNet import build_net
# # from models.FSNet_UDPNet import build_net  <-- 注释掉这行
# from models.ConvIR_UDPNet import build_net   # <-- 换成这行
torch.backends.cudnn.enabled = False



def _eval(model, args):
    state_dict = torch.load(args.test_model, weights_only=False)['state_dict']
    #state_dict = torch.load(args.test_model)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    factor = 8 #32

    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            # input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            #input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect'
            # CPU 上执行填充
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            # 填充完后再送入 5090
            input_img = input_img.to(device)
            # --------------------------------------------

            tm = time.time()

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            label_img = (label_img).cuda()
            psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))),
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False)
            print('%d iter PSNR_dehazing: %.3f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
            ssim_adder(ssim_val)

            # if args.save_image:
            #     save_name = os.path.join(args.result_dir, name[0])
            #     pred_clip += 0.5 / 255
            #     pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            #     pred.save(save_name)
            #     # # 1. 去掉原始后缀（如 .JPG），只保留文件名（如 0001）
            #     # file_basename = os.path.splitext(name[0])[0]
            #     # # 2. 重新拼接路径，后缀改为 .png
            #     # save_name = os.path.join(args.result_dir, file_basename + '.png')
            #     # # 3. 保存图片
            #     # # 注意：我删掉了原来的 pred_clip += 0.5 / 255
            #     # # 因为保存 PNG 是无损的，不需要那个偏置，去掉后画面对比度会稍微好一点点
            #     # pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            #     # pred_img.save(save_name)
            #     torch.cuda.empty_cache()
            if args.save_image:
                # 1. 强制替换后缀为 .png，保证无损保存，且符合官方评测脚本要求
                file_basename = os.path.splitext(name[0])[0]
                save_name = os.path.join(args.result_dir, file_basename + '.png')

                # 2. 极限保 PSNR 神技：加上 0.5/255 实现四舍五入，对抗 PyTorch 的向下取整
                pred_clip += 0.5 / 255.0

                # 3. 转换为 PIL Image 并保存
                pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_img.save(save_name, format='PNG')
                # ... 保存图片后 .
                torch.cuda.empty_cache()

            psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr_val)

            print('%d iter PSNR: %.3f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

        print('==========================================================')
        print('The average PSNR is %.3f dB' % (psnr_adder.average()))
        print('The average SSIM is %.5f dB' % (ssim_adder.average()))

        print("Average time: %f" % adder.average())

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    # print(model)
    # --- 加入这一行，针对 5090 进行自动编译加速 ---
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
    # -------------------------------------------

    if torch.cuda.is_available():
        model.cuda()
    _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='FSNet', type=str)
    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/data1t/dehaze/Haze4K')
    # Test
    parser.add_argument('--test_model', type=str, default='/home/zzy/DepthDehaze/Haze4K/results/spatialencoderrgb/test/epoch_epoch=01_psnr_val_psnr=33.89.ckpt')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'test')
    os.makedirs(args.result_dir, exist_ok=True)
    results_file = os.path.join(os.path.join('results/', args.model_name), 'results.txt')
    # sys.stdout = open(results_file, 'w')
    # sys.stderr = sys.stdout
    print(args)
    main(args)

#
# import os
# import time
# import torch
# import argparse
# import sys
# from utils import Adder
# from pytorch_msssim import ssim
# import torch.nn.functional as f
# from data import test_dataloader
# from torch.backends import cudnn
# from torchvision.transforms import functional as F
# from skimage.metrics import peak_signal_noise_ratio
# from models.FSNet_UDPNet import build_net
#
# # 5090 兼容性设置
# torch.backends.cudnn.enabled = False
#
#
# def transform(img, mode):
#     """ 对 4通道 (RGBD) 或 3通道 (RGB) 进行 8 种几何变换 """
#     if mode == 0: return img
#     if mode == 1: return torch.flip(img, [2])  # 水平翻转
#     if mode == 2: return torch.flip(img, [3])  # 垂直翻转
#     if mode == 3: return torch.flip(img, [2, 3])  # 水平+垂直翻转
#     if mode == 4: return img.transpose(2, 3)  # 转置 (H, W 互换)
#     if mode == 5: return torch.flip(img.transpose(2, 3), [2])
#     if mode == 6: return torch.flip(img.transpose(2, 3), [3])
#     if mode == 7: return torch.flip(img.transpose(2, 3), [2, 3])
#
#
# def inv_transform(img, mode):
#     """ 将推理结果反向旋转/翻转回来 """
#     if mode == 0: return img
#     if mode == 1: return torch.flip(img, [2])
#     if mode == 2: return torch.flip(img, [3])
#     if mode == 3: return torch.flip(img, [2, 3])
#     if mode == 4: return img.transpose(2, 3)
#     if mode == 5: return torch.flip(img, [2]).transpose(2, 3)
#     if mode == 6: return torch.flip(img, [3]).transpose(2, 3)
#     if mode == 7: return torch.flip(img, [2, 3]).transpose(2, 3)
#
#
# def _eval(model, args):
#     # 加载权重，适配 PyTorch 2.6
#     state_dict = torch.load(args.test_model, map_location='cpu', weights_only=False)['state_dict']
#     new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict, strict=False)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
#     torch.cuda.empty_cache()
#     adder = Adder()
#     model.eval()
#     factor = 8
#
#     with torch.no_grad():
#         psnr_adder = Adder()
#         ssim_adder = Adder()
#
#         for iter_idx, data in enumerate(dataloader):
#             input_img, label_img, name = data
#
#             # 1. 记录原始尺寸
#             h, w = input_img.shape[2], input_img.shape[3]
#
#             # --- Self-Ensemble 8次推理循环 ---
#             tm = time.time()
#             ensemble_pred = 0.0
#
#             print(f"正在处理图像 [{iter_idx + 1}]: {name[0]} (8次增强推理中...)")
#
#             for mode in range(8):
#                 # 变换输入 (CPU上做变换)
#                 aug_input = transform(input_img, mode)
#
#                 # 计算 Padding 尺寸 (针对当前变换后的 H, W)
#                 curr_h, curr_w = aug_input.shape[2], aug_input.shape[3]
#                 H, W = ((curr_h + factor) // factor) * factor, ((curr_w + factor) // factor * factor)
#                 padh, padw = H - curr_h, W - curr_w
#
#                 # CPU Padding 并送入 GPU
#                 aug_input = f.pad(aug_input, (0, padw, 0, padh), 'reflect').to(device)
#
#                 # 推理
#                 out = model(aug_input)[2]
#
#                 # 裁剪 Padding 并反向变换回来
#                 out = out[:, :, :curr_h, :curr_w]
#                 ensemble_pred += inv_transform(out.cpu(), mode)
#
#                 # 清理显存碎片
#                 torch.cuda.empty_cache()
#
#             # 2. 求 8 次结果的平均值
#             pred = ensemble_pred / 8.0
#             # -------------------------------
#
#             elapsed = time.time() - tm
#             adder(elapsed)
#
#             pred_clip = torch.clamp(pred, 0, 1)
#             pred_numpy = pred_clip.squeeze(0).numpy()
#             label_numpy = label_img.squeeze(0).numpy()
#
#             label_img = label_img.to(device)
#             pred_clip = pred_clip.to(device)
#
#             psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
#             print('%d iter PSNR: %.3f time: %.3f' % (iter_idx + 1, psnr_val, elapsed))
#             psnr_adder(psnr_val)
#
#             if args.save_image:
#                 file_basename = os.path.splitext(name[0])[0]
#                 save_name = os.path.join(args.result_dir, file_basename + '.png')
#
#                 # 极限保 PSNR：四舍五入补偿
#                 pred_save = pred_clip.clone()
#                 pred_save += 0.5 / 255.0
#                 pred_save = torch.clamp(pred_save, 0, 1)
#
#                 pred_img = F.to_pil_image(pred_save.squeeze(0).cpu(), 'RGB')
#                 pred_img.save(save_name, format='PNG')
#                 torch.cuda.empty_cache()
#
#         print('==========================================================')
#         print('The average PSNR is %.3f dB' % (psnr_adder.average()))
#         print("Average time: %f" % adder.average())
#
#
# def main(args):
#     cudnn.benchmark = True
#     if not os.path.exists(args.result_dir):
#         os.makedirs(args.result_dir, exist_ok=True)
#
#     model = build_net()
#     if torch.cuda.is_available():
#         model.cuda()
#     _eval(model, args)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', default='FSNet_SelfEnsemble', type=str)
#     parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
#     parser.add_argument('--data_dir', type=str, default='')
#     parser.add_argument('--test_model', type=str, default='')
#     parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
#
#     args = parser.parse_args()
#     args.result_dir = os.path.join('results/', args.model_name, 'test_ensemble')
#     os.makedirs(args.result_dir, exist_ok=True)
#     main(args)
