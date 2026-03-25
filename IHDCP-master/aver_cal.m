% % 1. 配置路径（请确保路径末尾没有多余空格）
% path1 = 'D:\matlab2024\cccode\IHDCP-master\udpOrigin';
% path2 = 'D:\matlab2024\cccode\IHDCP-master\dehazeformer_8enhance';
% save_path = 'D:\matlab2024\cccode\IHDCP-master\udpO+de8';
% 
% % 如果目录不存在则创建
% if ~exist(save_path, 'dir')
%     mkdir(save_path);
% end
% 
% % 2. 获取文件列表
% fileList = dir(fullfile(path1, '*.png'));
% numFiles = length(fileList);
% 
% if numFiles == 0
%     error('在文件夹1中没有找到 .png 文件，请检查路径。');
% end
% 
% fprintf('开始处理，共 %d 张图片...\n', numFiles);
% 
% % 3. 循环处理
% for i = 1:numFiles
%     fileName = fileList(i).name;
%     fullPath1 = fullfile(path1, fileName);
%     fullPath2 = fullfile(path2, fileName);
% 
%     % 检查文件夹2是否存在同名文件
%     if exist(fullPath2, 'file')
%         % 读取图片
%         img1 = imread(fullPath1);
%         img2 = imread(fullPath2);
% 
%         % 转为 double 进行计算（防止 uint8 相加溢出）
%         d1 = double(img1);
%         d2 = double(img2);
% 
%         % --- 修正后的尺寸检查 ---
%         % 使用 isequal 比较整个 size 向量（包括长、宽、通道数）
%         if ~isequal(size(d1), size(d2))
%             % 如果尺寸不匹配，将 img2 缩放到 img1 的大小
%             d2 = imresize(d2, [size(d1, 1), size(d1, 2)]);
% 
%             % 处理颜色通道不匹配的情况（如一张灰度一张彩色）
%             if size(d1, 3) ~= size(d2, 3)
%                 if size(d1, 3) == 3 && size(d2, 3) == 1
%                     d2 = cat(3, d2, d2, d2); % 灰度转彩色
%                 elseif size(d1, 3) == 1 && size(d2, 3) == 3
%                     d2 = rgb2gray(uint8(d2)); % 彩色转灰度
%                     d2 = double(d2);
%                 end
%             end
%         end
% 
%         % 4. 求平均值
%         avg_img = (d1 + d2) / 2;
% 
%         % 转回 uint8 格式保存
%         imwrite(uint8(avg_img), fullfile(save_path, fileName));
% 
%         if mod(i, 10) == 0 % 每10张显示一次进度
%             fprintf('已完成: %d/%d\n', i, numFiles);
%         end
%     else
%         fprintf('跳过: %s（在文件夹2中未找到）\n', fileName);
%     end
% end
% 
% fprintf('全部处理完成！结果已存至: %s\n', save_path);




% 1. 配置三个源路径
path1 = 'D:\matlab2024\cccode\IHDCP-master\udp8enhance';
path2 = 'D:\matlab2024\cccode\IHDCP-master\ipc_8enhance';
path3 = 'D:\matlab2024\cccode\IHDCP-master\dehazeformer_8enhance';

% 配置保存路径
save_path = 'D:\matlab2024\cccode\IHDCP-master\3way8_results';

% 如果保存目录不存在则创建
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

% 2. 获取第一个文件夹的文件列表作为基准
fileList = dir(fullfile(path1, '*.png'));
numFiles = length(fileList);

if numFiles == 0
    error('在文件夹1中没有找到 .png 文件，请检查路径。');
end

fprintf('开始处理，共 %d 组图片...\n', numFiles);

% 3. 循环处理
for i = 1:numFiles
    fileName = fileList(i).name;
    
    % 构造三个文件的完整路径
    f1 = fullfile(path1, fileName);
    f2 = fullfile(path2, fileName);
    f3 = fullfile(path3, fileName);
    
    % 检查另外两个文件夹是否存在同名文件
    if exist(f2, 'file') && exist(f3, 'file')
        % 读取图片
        img1 = imread(f1);
        img2 = imread(f2);
        img3 = imread(f3);
        
        % 转为 double 精度进行计算，防止 uint8 相加超过 255 溢出
        d1 = double(img1);
        d2 = double(img2);
        d3 = double(img3);
        
        % --- 尺寸对齐 (以第一张图为基准) ---
        sz1 = size(d1);
        
        % 检查并调整第二张图尺寸
        if ~isequal(size(d2), sz1)
            d2 = imresize(d2, [sz1(1), sz1(2)]);
            % 如果通道数不匹配（如灰度vs彩色），尝试对齐
            if size(d2, 3) ~= sz1(3)
                if sz1(3) == 3, d2 = cat(3, d2, d2, d2); else, d2 = rgb2gray(uint8(d2)); d2 = double(d2); end
            end
        end
        
        % 检查并调整第三张图尺寸
        if ~isequal(size(d3), sz1)
            d3 = imresize(d3, [sz1(1), sz1(2)]);
            if size(d3, 3) ~= sz1(3)
                if sz1(3) == 3, d3 = cat(3, d3, d3, d3); else, d3 = rgb2gray(uint8(d3)); d3 = double(d3); end
            end
        end
        
        % --- 4. 求三者平均值 ---
        avg_img = (d1 + d2 + d3) / 3.0;
        
        % 转回 uint8 并保存
        imwrite(uint8(avg_img), fullfile(save_path, fileName));
        
        % 进度显示
        if mod(i, 20) == 0
            fprintf('已完成: %d/%d\n', i, numFiles);
        end
    else
        fprintf('跳过: %s (在文件夹2或3中未找到同名文件)\n', fileName);
    end
end

fprintf('全部处理完成！结果已存至: %s\n', save_path);