%% 从文本文件导入数据。
% 用于从以下文本文件导入数据的脚本：
%
%    H:\ETH_Fotokite\AGZ\Log Files\GroundTruthAGL.csv
%
% 若要将代码扩展到不同的选定数据或不同的文本文件，请生成函数而不是脚本。

% 由 MATLAB 于 2015/10/20 16:37:39 自动生成

%% 初始化变量。
filename = 'H:\ETH_Fotokite\AGZ\Log Files\GroundTruthAGL.csv';
delimiter = ',';
startRow = 2;

%% 每行文本的格式字符串：
%   第1列: 双精度浮点数 (%f)
%	第2列: 双精度浮点数 (%f)
%   第3列: 双精度浮点数 (%f)
%	第4列: 双精度浮点数 (%f)
%   第5列: 双精度浮点数 (%f)
%	第6列: 双精度浮点数 (%f)
%   第7列: 双精度浮点数 (%f)
%	第8列: 双精度浮点数 (%f)
%   第9列: 双精度浮点数 (%f)
%	第10列: 双精度浮点数 (%f)
% 有关详细信息，请参阅 TEXTSCAN 文档。
formatSpec = '%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% 打开文本文件。
fileID = fopen(filename,'r');

%% 根据格式字符串读取数据列。
% 此调用基于用于生成此代码的文件结构。如果其他文件出现问题，请尝试从“导入工具”重新生成代码。
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);

%% 关闭文本文件。
fclose(fileID);

%% 无法导入数据的后期处理。
% 导入过程中未应用任何无法导入数据的规则，因此不包含后期处理代码。
% 要生成适用于不可导入数据的代码，请在文件中选择不可导入的单元格并重新生成脚本。

%% 将导入的数组分配给列变量名
imgid = dataArray{:, 1};
x_gt = dataArray{:, 2};
y_gt = dataArray{:, 3};
z_gt = dataArray{:, 4};
omega_gt = dataArray{:, 5};
phi_gt = dataArray{:, 6};
kappa_gt = dataArray{:, 7};
x_gps = dataArray{:, 8};
y_gps = dataArray{:, 9};
z_gps = dataArray{:, 10};

%% 清除临时变量
clearvars filename delimiter startRow formatSpec fileID dataArray ans;