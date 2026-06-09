%% 并行-串行信念规则库 (PS-BRB) 主优化程序
clc; clear all;

global TrainData TrainLabels TestData TestLabels
global L1 L2 L3 N M_sub M3

%% ======= 1. 参数结构定义 (严丝合合缝凑出 96 维拓扑) =======
L1 = 4;      % BRB1 (显性组子库) 规则数
L2 = 4;      % BRB2 (隐性组子库) 规则数
L3 = 10;     % BRB3 (串行融合层) 规则数
N = 4;       % 结果健康等级数 (1, 2, 3, 4)
M_sub = 2;   % 并行层子库的特征输入数 (各挑2个最核心属性)
M3 = 2;      % 串行层子库的输入属性数 (接收BRB1和BRB2的输出)

%% ======= 2. 读取带表头的数据集并进行降维横向组合 =======
X1_train = readmatrix('brb1_train.txt');
X1_test  = readmatrix('brb1_test.txt');
X2_train = readmatrix('brb2_train.txt');
X2_test  = readmatrix('brb2_test.txt');

% 降维横向组合：TrainData总共包含4列特征（1:2给BRB1，3:4给BRB2）
TrainData   = [X1_train(:, 1:2), X2_train(:, 1:2)]; 
TrainLabels = X1_train(:, 7); % 提取标签

TestData    = [X1_test(:, 1:2), X2_test(:, 1:2)];   
TestLabels  = X1_test(:, 7);  % 提取标签

%% ======= 3. 协同优化参数精确初始化 =======
Param_BRB1 = L1 * N + L1 + M_sub;  % 4*4 + 4 + 2 = 22
Param_BRB2 = L2 * N + L2 + M_sub;  % 4*4 + 4 + 2 = 22
Param_BRB3 = L3 * N + L3 + M3;     % 10*4 + 10 + 2 = 52

Total_Params = Param_BRB1 + Param_BRB2 + Param_BRB3; % 严格等于 96 维

x0 = ones(Total_Params, 1) * 0.5; 
lb = zeros(Total_Params, 1);
ub = ones(Total_Params, 1);

disp(['当前精确凑出的总优化参数量为: ', num2str(length(x0)), ' 维']);
G = 800; % CMA-ES 迭代次数

% 强制转化为标准一维列向量传入算法，确保高版本与低版本矩阵兼容量
x0_input = x0(:);
lb_input = lb(:);
ub_input = ub(:);

%% ======= 4. 调用 cmaes 算法 =======
disp('正在进行并行-串行 BRB 参数联合优化中...');
best_x = cmaes(x0_input, G, [], [], ub_input, lb_input, 1);

%% ======= 5. 测试集验证 =======
disp('优化完成！正在检验测试集性能...');
[test_rmse, yp] = fun_test(best_x, 1);

fprintf('\n================ 实验结果 ================ \n');
fprintf('PS-BRB 并行-串行融合模型最终测试集 RMSE 误差: %.4f\n', test_rmse);

true_labels = TestLabels;
predicted_labels = round(yp);
save('eval_res.mat', 'true_labels', 'predicted_labels');