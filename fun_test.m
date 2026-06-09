function [f, Final_Outputs] = fun_test(x, sub)
global TestData TestLabels
global L1 L2 L3 N M_sub M3

T = size(TestData, 1);
if size(x, 1) == 1; x = x'; end

idx = 1;
beta1 = reshape(x(idx : idx + L1*N - 1), [N, L1])'; idx = idx + L1*N;
RuleWT1 = x(idx : idx + L1 - 1); idx = idx + L1;
AttrWT1 = x(idx : idx + M_sub - 1); idx = idx + M_sub;

beta2 = reshape(x(idx : idx + L2*N - 1), [N, L2])'; idx = idx + L2*N;
RuleWT2 = x(idx : idx + L2 - 1); idx = idx + L2;
AttrWT2 = x(idx : idx + M_sub - 1); idx = idx + M_sub;

beta3 = reshape(x(idx : idx + L3*N - 1), [N, L3])'; idx = idx + L3*N;
RuleWT3 = x(idx : idx + L3 - 1); idx = idx + L3;
AttrWT3 = x(idx : idx + M3 - 1); idx = idx + M3;

y_sub_ref = [1, 2, 3, 4]; y3_ref = [1, 2, 3, 4]; Doutput = [1, 2, 3, 4]; 
Final_Outputs = zeros(T, 1);

%% 逐样本测试
for n = 1:T
    % 🌟【核心修正】TestData 只有 4 列
    feat_brb1 = TestData(n, 1:2); 
    feat_brb2 = TestData(n, 3:4);
    
    out_brb1 = er_engine(feat_brb1, beta1, RuleWT1, AttrWT1, y_sub_ref, Doutput);
    out_brb2 = er_engine(feat_brb2, beta2, RuleWT2, AttrWT2, y_sub_ref, Doutput);
    
    feat_brb3 = [out_brb1, out_brb2];
    Final_Outputs(n) = er_engine(feat_brb3, beta3, RuleWT3, AttrWT3, y3_ref, Doutput);
end
f = sqrt(mean((Final_Outputs - TestLabels).^2));
end

%% ================== 通用 ER 推理机引擎 ==================
function y_hat = er_engine(InputFeat, beta, RuleWT, AttrWT, x_ref, Doutput)
    L = size(beta, 1); N_class = size(beta, 2); M = length(InputFeat);
    Alpha = zeros(M, length(x_ref));
    for m = 1:M
        val = InputFeat(m);
        for k = 1:length(x_ref)-1
            if val >= x_ref(k) && val <= x_ref(k+1)
                Alpha(m, k+1) = (val - x_ref(k)) / (x_ref(k+1) - x_ref(k));
                Alpha(m, k) = 1 - Alpha(m, k+1);
            end
        end
        if val < x_ref(1), Alpha(m, 1) = 1; end
        if val > x_ref(end), Alpha(m, end) = 1; end
    end
    AM = ones(L, 1);
    for k = 1:L
        for m = 1:M
            AM(k) = AM(k) * (Alpha(m, 1))^AttrWT(m);
        end
        AM(k) = AM(k) * RuleWT(k);
    end
    AU = sum(AM); if AU == 0; ActivationW = ones(L, 1)/L; else ActivationW = AM / AU; end
    Sum1 = sum(beta, 2); temp1 = ones(1, N_class);
    for j = 1:N_class
        for k = 1:L
            Belief(k,j) = ActivationW(k) * beta(k,j) + 1 - ActivationW(k) * Sum1(k);
            temp1(j) = temp1(j) * Belief(k,j);
        end
    end
    temp2 = sum(temp1); temp3 = 1;
    for k = 1:L; temp3 = temp3 * (1 - ActivationW(k) * Sum1(k)); end
    Combined_Beta = zeros(1, N_class);
    denom = temp2 - (N_class-1)*temp3 - prod(1 - ActivationW.*Sum1);
    if denom == 0, y_hat = mean(Doutput); return; end
    for j = 1:N_class; Combined_Beta(j) = (temp1(j) - temp3) / denom; end
    y_hat = sum(Combined_Beta .* Doutput);
end