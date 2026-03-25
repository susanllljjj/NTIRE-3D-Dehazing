import os
import torch
import argparse
from torch.backends import cudnn
#from models.FSNet import build_net
from models.FSNet_UDPNet import build_net
from trainlightning import _train

def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net()
    # print(model)
    # --- 新增：加载预训练权重进行微调 ---
    if args.test_model and os.path.exists(args.test_model):
        print(f"正在从 {args.test_model} 加载预训练权重进行微调...")
        #checkpoint = torch.load(args.test_model, map_location='cpu')
        # 显式设置 weights_only=False 允许加载旧版本的完整对象
        checkpoint = torch.load(args.test_model, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # 移除 PyTorch Lightning 自动添加的 "model." 前缀
            new_key = k.replace("model.", "")
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("--- 权重加载成功 ---")
    # --------------------------------

    if torch.cuda.is_available():
        model.cuda()
        
    if args.mode == 'train':
        _train(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='ConvIR', type=str)
    parser.add_argument('--data_dir', type=str, default='')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=3000)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--logs', type=str, default='')

    # Test
    parser.add_argument('--test_model', type=str, default='gopro.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'checkpoints')
    args.result_dir = os.path.join('results/', args.model_name, 'Haze4k')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/FSNet.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    # print(args)
    main(args)
