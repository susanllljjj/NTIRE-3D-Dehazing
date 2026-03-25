% 1. 配置路径
src_path = 'D:\matlab2024\cccode\IHDCP-master\test_avg';
dst_path = 'D:\matlab2024\cccode\IHDCP-master\test_avg_renamed';

% 如果目标目录不存在则创建
if ~exist(dst_path, 'dir')
    mkdir(dst_path);
end

% 2. 获取所有 png 文件
fileList = dir(fullfile(src_path, '*.png'));
numFiles = length(fileList);

if numFiles == 0
    error('源目录中没有找到 .png 文件。');
end

% 3. 提取所有前缀并分组
% 我们先遍历一遍，记录每个前缀对应的文件名
allNames = {fileList.name};
prefixes = cell(1, numFiles);

for i = 1:numFiles
    nameSplit = split(allNames{i}, '_');
    if length(nameSplit) < 2
        fprintf('警告: 文件 %s 格式不符合 "前缀_数字.png"，已跳过。\n', allNames{i});
        prefixes{i} = '';
        continue;
    end
    prefixes{i} = nameSplit{1}; % 提取前缀，如 Futaba
end

% 获取唯一的前缀列表（去重并排序）
uniquePrefixes = unique(prefixes(~cellfun(@isempty, prefixes)));

fprintf('检测到 %d 个不同的场景前缀。\n', length(uniquePrefixes));

% 4. 按前缀处理和重新编号
for p = 1:length(uniquePrefixes)
    currentPrefix = uniquePrefixes{p};
    % 找出属于当前前缀的所有文件
    idx = find(strcmp(prefixes, currentPrefix));
    groupFiles = allNames(idx);
    
    % 重要：对组内文件进行自然排序（防止 10.png 排在 2.png 前面）
    % MATLAB 默认的 sort 对字符串是按照 ASCII 排序的，通常对于 0024 这种带补位的格式没问题
    groupFiles = sort(groupFiles);
    
    fprintf('正在处理前缀: %s (共 %d 张)... ', currentPrefix, length(groupFiles));
    
    % 遍历组内文件并重新命名
    for k = 1:length(groupFiles)
        oldName = groupFiles{k};
        
        % 生成新名字：前缀转小写 + 下划线 + 4位补零数字
        % eg: futaba_0001.png
        newName = sprintf('%s_%04d.png', lower(currentPrefix), k);
        
        % 复制文件到新目录
        srcFile = fullfile(src_path, oldName);
        dstFile = fullfile(dst_path, newName);
        copyfile(srcFile, dstFile);
    end
    fprintf('完成。\n');
end

fprintf('\n所有文件已重命名并保存至: %s\n', dst_path);