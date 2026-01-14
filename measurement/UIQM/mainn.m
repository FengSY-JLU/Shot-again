% 设置根文件夹路径
rootFolder = '../clg60Results+SPBb+SGCA+margin0.2a/J/';

% 获取根文件夹下所有子文件夹路径
subfolders = dir(rootFolder);
subfolderNames = {subfolders([subfolders.isdir]).name}; % 获取子文件夹名称

% 存储结果的结构体数组
results = struct('folderNumber', {}, 'mean_uiqm', {}, 'mean_uicm', {});

% 遍历每个子文件夹
for i = 1:length(subfolderNames)
    folderName = subfolderNames{i};
    % 提取文件夹路径中末尾的数字
    folderNumber = extractFolderNumber(folderName);
    if ~isempty(folderNumber)
        % 构建完整的文件夹路径
        folderPath = fullfile(rootFolder, folderName);

        % 输出文件夹路径中末尾的数字
        disp(['Processing folder with number: ' num2str(folderNumber)]);
        
        % 获取文件夹中所有PNG文件
        Files = dir(fullfile(folderPath, '*.png'));
        number = length(Files);
        score_uiqm = zeros(1, number);
        score_uicm = zeros(1, number);

        % 计算每个图像的UIQM和UICM分数
        for j = 1:number
            filename = Files(j).name;
            img = imread(fullfile(folderPath, filename));
            uiqm = UIQM(img);
            uiqm = roundn(uiqm, -4);
            score_uiqm(j) = uiqm;
            uicm = UICM(img);
            uicm = roundn(uicm, -4);
            score_uicm(j) = uicm; 
        end

        % 计算UIQM和UICM分数的平均值
        mean_uiqm = mean(score_uiqm);
        mean_uicm = mean(score_uicm);

        % 存储结果
        results(end+1).folderNumber = folderNumber;
        results(end).mean_uiqm = mean_uiqm;
        results(end).mean_uicm = mean_uicm;
    end
end

% 输出结果到result.txt文件
outputFile = fopen('result.txt', 'w');
fprintf(outputFile, 'Folder\tMean UIQM\tMean UICM\n');
for i = 1:length(results)
    fprintf(outputFile, '%d\t%.4f\t%.4f\n', results(i).folderNumber, results(i).mean_uiqm, results(i).mean_uicm);
end
fclose(outputFile);
disp('Results saved to result.txt');

% 输出结果
disp('Results:');
for i = 1:length(results)
    disp(['Folder ' num2str(results(i).folderNumber) ':']);
    disp(['Mean UIQM: ' num2str(results(i).mean_uiqm)]);
    disp(['Mean UICM: ' num2str(results(i).mean_uicm)]);
end

% 定义函数：提取文件夹路径中末尾的数字
function folderNumber = extractFolderNumber(folderName)
    % 使用正则表达式提取末尾的数字
    matches = regexp(folderName, '\d+$', 'match', 'once');
    if ~isempty(matches)
        folderNumber = str2double(matches);
    else
        folderNumber = [];
    end
end

