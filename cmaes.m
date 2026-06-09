function bestx=cmaes(x,G,Aeq,beq,ub,lb,sub)
% (mu/mu_w, lambda)-CMA-ES 优化引擎（完全自适应自整定版）

% --------------------  Initialization --------------------------------
x = x(:);              % 确保传入初始解是标准列向量
N = length(x);         % 🌟【核心修正】自动跟踪获取参数总维数，支持自适应结构切换

ub = ub(:);            % 强制转化为列，避免高低版本转置歧义
lb = lb(:);

xmean = x;              
sigma = 0.5;          
counteval = 0;        
stopMaxEval = G * 100; 

lambda = 4 + floor(3*log(N));  
mu = lambda/2;                 
weights = log(mu+1/2)-log(1:floor(mu)); 
mu = floor(mu);        

% 🌟【核心修正】强行将权重压成标准列向量 [mu, 1]，彻底杜绝 * 矩阵乘法不匹配问题
weights = weights(:) / sum(weights);     

mueff=sum(weights)^2/sum(weights.^2); 

cc = (4 + mueff/N) / (N+4 + 2*mueff/N);  
cs = (mueff+2) / (N+mueff+5);  
c1 = 2 / ((N+1.3)^2+mueff);    
cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+2*mueff/2));  
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; 

pc = zeros(N,1); ps = zeros(N,1);   
B = eye(N,N);                       
D = ones(N,1);                      
C = B * diag(D.^2) * B';            
invsqrtC = B * diag(D.^-1) * B';    
eigeneval = 0;                      
chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  

best_fit = Inf;
bestx = x;

% -------------------- Generation Loop --------------------------------
for gen = 1:G
    arx = zeros(N, lambda); 
    arfitness = zeros(1, lambda);
    
    for k=1:lambda
        arx(:,k) = xmean + sigma * (B * (D .* randn(N,1))); 
        
        % 约束边界控制控制
        for kk=1:N
            if arx(kk,k) > ub(kk); arx(kk,k) = ub(kk); end
            if arx(kk,k) < lb(kk); arx(kk,k) = lb(kk); end
        end
        
        arfitness(k) = fun(arx(:,k)', sub); 
    end
    counteval = counteval + lambda;
    
    % 排序选择最佳个体
    [arfitness, arindex] = sort(arfitness); 
    xold = xmean;
    
    % 🌟【安全执行】由于 weights 被纠正为了 [mu,1]，此时 [96,mu]*[mu,1] 会完美运算得到 [96,1]
    xmean = arx(:,arindex(1:mu)) * weights; 
    
    if arfitness(1) < best_fit
        best_fit = arfitness(1);
        bestx = arx(:, arindex(1));
    end
    
    % 周期性在控制台打印训练进度
    if mod(gen, 10) == 0 || gen == 1
        fprintf('迭代代数: %d / %d, 当前模型最佳均方根误差 (RMSE): %.4f\n', gen, G, best_fit);
    end
    
    % 进化路径与参数自适应迭代
    ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma; 
    hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4 + 2/(N+1);
    pc = (1-cc)*pc + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
    
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu)); 
    C = (1-c1-cmu) * C + c1 * (pc * pc' + (1-hsig) * cc*(2-cc) * C) + cmu * artmp * diag(weights) * artmp';
    
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
    
    if counteval - eigeneval > lambda/(c1+cmu)/N/10
        eigeneval = counteval;
        C = triu(C) + triu(C,1)'; 
        [B,D] = eig(C);           
        D = sqrt(diag(D));        
        invsqrtC = B * diag(D.^-1) * B';
    end
end
end