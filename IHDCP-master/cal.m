%% IHDCP 性能评估脚本 (针对特定命名规则)
clear; clc;

% 1. 设置路径
gt_dir = 'gt_images/';      % GT 图像文件夹 (.JPG)
out_dir = 'output_images/'; % 模型结果文件夹 (.png)

% 2. 获取 GT 文件列表 (只获取 .JPG 文件)
gt_files = dir(fullfile(gt_dir, '*.png')); 
% 如果你的后缀是大写的 .JPG 就用 *.JPG，如果是小写就用 *.jpg

if isempty(gt_files)
    error('在 gt_images 文件夹中没有找到 .JPG 文件！请检查后缀大小写。');
end

% 3. 初始化变量
num_images = length(gt_files);
all_psnr = zeros(num_images, 1);
all_ssim = zeros(num_images, 1);
valid_count = 0; % 记录成功匹配的数量

fprintf('开始计算指标...\n');
fprintf('------------------------------------\n');

% 4. 循环处理
for i = 1:num_images
    % A. 获取 GT 文件名和基础名称 (例如从 "0001.JPG" 提取出 "0001")
    gt_full_name = gt_files(i).name;
    [~, base_name, ~] = fileparts(gt_full_name);
    
    % B. 构建对应的输出文件名 (例如 "0001_processed.png")
    % out_full_name = [base_name, '_processed.png'];
    out_full_name = [base_name, '.png'];
    out_path = fullfile(out_dir, out_full_name);
    gt_path = fullfile(gt_dir, gt_full_name);
    
    % C. 检查输出文件是否存在
    if exist(out_path, 'file')
        % 读取图像
        img_gt = imread(gt_path);
        img_out = imread(out_path);
        
        % 统一尺寸 (以 GT 为准，防止因算法导致的长宽 1 像素偏差)
        if size(img_gt, 1) ~= size(img_out, 1) || size(img_gt, 2) ~= size(img_out, 2)
            img_out = imresize(img_out, [size(img_gt, 1), size(img_gt, 2)]);
        end
        
        % 计算指标
        valid_count = valid_count + 1;
        all_psnr(valid_count) = psnr(img_out, img_gt);
        all_ssim(valid_count) = ssim(img_out, img_gt);
        
        fprintf('[%d/%d] 匹配成功: %s <-> %s | PSNR: %.2f\n', ...
            i, num_images, gt_full_name, out_full_name, all_psnr(valid_count));
    else
        fprintf('[%d/%d] 跳过: 找不到对应的输出文件 %s\n', i, num_images, out_full_name);
    end
end

% 5. 输出最终平均结果
if valid_count > 0
    final_psnr = mean(all_psnr(1:valid_count));
    final_ssim = mean(all_ssim(1:valid_count));
    
    fprintf('------------------------------------\n');
    fprintf('测试完成！有效样本数: %d\n', valid_count);
    fprintf('平均 PSNR: %.4f\n', final_psnr);
    fprintf('平均 SSIM: %.4f\n', final_ssim);
    fprintf('------------------------------------\n');
else
    fprintf('错误：没有匹配到任何图像对，请检查文件名规则。\n');
end