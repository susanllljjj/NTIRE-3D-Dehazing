% %% 路径设置
% basePath = 'D:\matlab2024\cccode\IHDCP-master\';
% dirDehaze = fullfile(basePath, 'dehazeA');   % .JPG
% dirUdp    = fullfile(basePath, 'udpA');      % .png
% dirGT     = fullfile(basePath, 'gt_images'); % .JPG
% 
% % 融合结果输出路径
% outBest = fullfile(basePath, 'weighted_fusion_best');
% if ~exist(outBest, 'dir'), mkdir(outBest); end
% 
% %% 1. 预加载数据到内存 (加速搜索)
% fprintf('正在预加载图片到内存...\n');
% fileList = dir(fullfile(dirDehaze, '*.JPG'));
% numFiles = length(fileList);
% 
% % 使用 cell 数组存储 double 类型图片
% dataD = cell(numFiles, 1);
% dataU = cell(numFiles, 1);
% dataG = cell(numFiles, 1);
% 
% for i = 1:numFiles
%     [~, stem, ~] = fileparts(fileList(i).name);
%     imgD = double(imread(fullfile(dirDehaze, [stem, '.JPG'])));
%     imgU = double(imread(fullfile(dirUdp,    [stem, '.png'])));
%     imgG = imread(fullfile(dirGT,     [stem, '.JPG']));
% 
%     % 以 GT 尺寸为准对齐
%     sz = size(imgG);
%     if any(size(imgD) ~= sz), imgD = imresize(imgD, sz(1:2)); end
%     if any(size(imgU) ~= sz), imgU = imresize(imgU, sz(1:2)); end
% 
%     dataD{i} = imgD;
%     dataU{i} = imgU;
%     dataG{i} = imgG;
% end
% 
% %% 2. 权重搜索 (Grid Search)
% weights = 0:0.05:1; % 从 0 到 1，步长 0.01
% bestPsnr = 0;
% bestW = 0;
% 
% fprintf('开始搜索最优权重...\n');
% fprintf('%-10s | %-10s\n', 'Weight(D)', 'Avg PSNR');
% fprintf('-----------------------\n');
% 
% for w = weights
%     currentTotalPsnr = 0;
% 
%     for i = 1:numFiles
%         % 加权计算: w*Dehaze + (1-w)*UDP
%         % 使用 round 保证像素级精准
%         fused = uint8(round(w * dataD{i} + (1-w) * dataU{i}));
%         currentTotalPsnr = currentTotalPsnr + psnr(fused, dataG{i});
%     end
% 
%     avgPsnr = currentTotalPsnr / numFiles;
%     fprintf('%.2f       | %.4f\n', w, avgPsnr);
% 
%     % 更新最优解
%     if avgPsnr > bestPsnr
%         bestPsnr = avgPsnr;
%         bestW = w;
%     end
% end
% 
% fprintf('-----------------------\n');
% fprintf('搜索完成！\n');
% fprintf('最优权重 (Dehaze): %.2f\n', bestW);
% fprintf('最优权重 (UDP):    %.2f\n', 1 - bestW);
% fprintf('最高平均 PSNR:     %.4f\n', bestPsnr);
% 
% %% 3. 使用最优权重生成并保存无损图片
% fprintf('正在使用最优权重生成无损 PNG...\n');
% for i = 1:numFiles
%     [~, stem, ~] = fileparts(fileList(i).name);
%     finalFused = uint8(round(bestW * dataD{i} + (1-bestW) * dataU{i}));
%     imwrite(finalFused, fullfile(outBest, [stem, '.png']));
% end
% 
% fprintf('所有图片已保存至: %s\n', outBest);


%% 路径设置 (保持不变)
basePath = 'D:\matlab2024\cccode\IHDCP-master\';
dirD = fullfile(basePath, 'dehazeA'); 
dirI = fullfile(basePath, 'ipcA');    
dirU = fullfile(basePath, 'udpA');    
dirG = fullfile(basePath, 'gt_images'); 

outBest = fullfile(basePath, 'weighted_fusion_three_best');
if ~exist(outBest, 'dir'), mkdir(outBest); end

%% 1. 预加载数据 (以 uint8 存储，节省内存)
fprintf('正在以 uint8 格式预加载数据...\n');
fileList = dir(fullfile(dirD, '*.JPG'));
numFiles = length(fileList);

% 使用 uint8 存储，内存占用极低
dataD = cell(numFiles, 1);
dataI = cell(numFiles, 1);
dataU = cell(numFiles, 1);
dataG = cell(numFiles, 1);

for i = 1:numFiles
    [~, stem, ~] = fileparts(fileList(i).name);
    imgG = imread(fullfile(dirG, [stem, '.JPG']));
    sz = size(imgG);
    
    % 读取并立即调整尺寸，保持 uint8
    imgD = imread(fullfile(dirD, [stem, '.JPG']));
    if any(size(imgD) ~= sz), imgD = imresize(imgD, sz(1:2)); end
    
    imgI = imread(fullfile(dirI, [stem, '.png']));
    if any(size(imgI) ~= sz), imgI = imresize(imgI, sz(1:2)); end
    
    imgU = imread(fullfile(dirU, [stem, '.png']));
    if any(size(imgU) ~= sz), imgU = imresize(imgU, sz(1:2)); end
    
    dataD{i} = imgD;
    dataI{i} = imgI;
    dataU{i} = imgU;
    dataG{i} = imgG;
end

%% 2. 权重搜索
step = 0.05; % 建议先用 0.05 跑通，没问题再改 0.01
bestPsnr = 0;
bestW = [0, 0, 0];

fprintf('开始搜索权重...\n');

for w1 = 0:step:1
    for w2 = 0:step:(1 - w1)
        w3 = 1 - w1 - w2;
        currentTotalPsnr = 0;
        
        for i = 1:numFiles
            % 仅在计算时临时转为 double
            fused = uint8(round(w1 * double(dataD{i}) + ...
                               w2 * double(dataI{i}) + ...
                               w3 * double(dataU{i})));
            currentTotalPsnr = currentTotalPsnr + psnr(fused, dataG{i});
        end
        
        avgPsnr = currentTotalPsnr / numFiles;
        fprintf('W: [%.2f, %.2f, %.2f] -> PSNR: %.4f\n', w1, w2, w3, avgPsnr);
        
        if avgPsnr > bestPsnr
            bestPsnr = avgPsnr;
            bestW = [w1, w2, w3];
        end
    end
end

% ... (后续保存 PNG 的代码与之前一致) ...
fprintf('------------------------------------------------------------\n');
fprintf('搜索完成！\n');
fprintf('最优权重组合:\n');
fprintf('  1. Dehaze: %.2f\n', bestW(1));
fprintf('  2. IPC:    %.2f\n', bestW(2));
fprintf('  3. UDP:    %.2f\n', bestW(3));
fprintf('最高平均 PSNR: %.4f\n', bestPsnr);

%% 3. 使用最优权重保存最终无损结果
fprintf('正在使用最优权重生成无损 PNG 文件...\n');
for i = 1:numFiles
    [~, stem, ~] = fileparts(fileList(i).name);
    
    % 使用最优权重合成
    finalFused = uint8(round(bestW(1) * dataD{i} + ...
                             bestW(2) * dataI{i} + ...
                             bestW(3) * dataU{i}));
                         
    imwrite(finalFused, fullfile(outBest, [stem, '.png']));
end

fprintf('所有图片已无损保存至: %s\n', outBest);