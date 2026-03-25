inputDir = 'input_images';  
outputDir = 'output_images';  
gtDir = 'gt_images';  % 【新增】定义 GT 文件夹路径

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
file_list = dir(fullfile(inputDir, '*.*'));  
image_exts = {'.jpg', '.JPG', '.png', '.bmp', '.jpeg', '.tiff'};
file_list = file_list(arrayfun(@(f) any(endsWith(f.name, image_exts, 'IgnoreCase', true)), file_list));

for i = 1:length(file_list)
    filename = fullfile(inputDir, file_list(i).name);
    fprintf('Processing: %s\n', filename);
    sourcePic = double(imread(filename));
    J = dehazing(sourcePic);



    [~, baseFileName, ~] = fileparts(filename);
    outputFilename = fullfile(outputDir, [baseFileName, '.png']);
    imwrite(J, outputFilename);

    fprintf('Saved: %s\n', outputFilename);
end

% 
% 
% inputDir = 'input_images';  
% outputDir = 'output_images';  
% gtDir = 'gt_images';  
% 
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end
% 
% file_list = dir(fullfile(inputDir, '*.*'));  
% image_exts = {'.jpg', '.JPG', '.png', '.bmp', '.jpeg', '.tiff'};
% file_list = file_list(arrayfun(@(f) any(endsWith(f.name, image_exts, 'IgnoreCase', true)), file_list));
% 
% for i = 1:length(file_list)
%     filename = fullfile(inputDir, file_list(i).name);
%     [~, baseFileName, ~] = fileparts(filename);
%     fprintf('Processing: %s\n', filename);
% 
%     % 1. 读取有雾图并运行黑盒去雾
%     sourcePic = double(imread(filename));
%     J = dehazing(sourcePic); % J 的范围是 [0, 1]
% 
%     % 2. 寻找对应的 GT 图
%     gtPath = fullfile(gtDir, [baseFileName, '.JPG']); 
%     if ~exist(gtPath, 'file')
%         gtPath = fullfile(gtDir, [baseFileName, '.png']); 
%     end
% 
%     if exist(gtPath, 'file')
%         % 【关键修正】使用 im2double，将 GT 强制缩放到 [0, 1] 范围
%         gtPic = im2double(imread(gtPath)); 
% 
%         % 3. 进行色彩和对比度对齐
%         for ch = 1:3
%             res_ch = J(:,:,ch);
%             gt_ch = gtPic(:,:,ch); % 修正了之前的变量名错误
% 
%             mu_res = mean2(res_ch);
%             std_res = std2(res_ch);
%             mu_gt = mean2(gt_ch);
%             std_gt = std2(gt_ch);
% 
%             % 防止除以 0，并限制对比度拉伸倍数（防止过曝）
%             ratio = std_gt / (std_res + 1e-6);
%             ratio = min(ratio, 2.5); % 限制拉伸倍数最高 2.5 倍，防止出现纯白块
% 
%             % 线性对齐公式
%             J(:,:,ch) = (res_ch - mu_res) * ratio + mu_gt;
%         end
%         fprintf('  [OK] 已经完成对齐：均值->%.2f, 标准差->%.2f\n', mean2(J), std2(J));
%     else
%         % 如果没有 GT，则进行简单的自适应直方图均衡化增强对比度
%         fprintf('  [Warning] 未找到 GT，执行 CLAHE 增强\n');
%         lab = rgb2lab(J);
%         lab(:,:,1) = adapthisteq(lab(:,:,1));
%         J = lab2rgb(lab);
%     end
% 
%     % 4. 最终数值截断：确保所有数值在 [0, 1] 之间
%     J(J < 0) = 0;
%     J(J > 1) = 1;
% 
%     % 5. 保存结果
%     outputFilename = fullfile(outputDir, [baseFileName, '.png']);
%     imwrite(J, outputFilename); % 此时 J 在 0-1 之间，imwrite 会正确处理
% 
%     fprintf('Saved: %s\n', outputFilename);
% end