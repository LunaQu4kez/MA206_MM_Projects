function [xm,fm,noi] = func(A,b,c)
%% 介绍
% 单纯形法求解标准形线性规划问题: min cx s.t. Ax=b x>=0
% 输入参数: c为目标函数系数, A为约束方程组系数矩阵, b为约束方程组常数项
% 输出参数: xm为最求解，fm为最优函数值，noi为迭代次数
%% 准备
format rat                                                                 %元素使用分数表示
[m,n] = size(A);                                                           %m约束条件个数, n决策变量数
v=nchoosek(1:n,m);                                                         %创建一个矩阵，该矩阵的行由每次从1:n中的n个元素取出k个取值的所有可能组合构成。矩阵 C 包含 m!/((n–m)! m!) 行和 m列
index_Basis=[];
%% 提取可行解所在列
for i=1:size(v,1)                                                          %size(v,1)为在n中取m个的种类
    if A(:,v(i,:))==eye(m)                                                 %在中心部位A中取v的第i种取法取出m列判断是否存在m*m大小的单位矩阵
        index_Basis=v(i,:);                                                %把%存在单位矩阵的取法保存在列表index_Basis中
    end
end
%% 提取非基变量的索引                                                                        
ind_Nonbasis = setdiff(1:n, index_Basis);                                  %非基变量的索引,返回在1:n中出现而不在index_Basis（即基变量索引中出现的元素），并从小到大排序
noi=0;                          
while true
    x0 = zeros(n,1);
    x0(index_Basis) = b;                                                   %初始基可行解
    cB = c(index_Basis);                                                   %计算cB，即目标函数在基变量处对应的系数
    Sigma = zeros(1,n);                                                    %Sigma为检验数向量
    Sigma(ind_Nonbasis) = c(ind_Nonbasis)' - cB'*A(:,ind_Nonbasis);        %计算检验数（非基变量），因为基变量对应的初始检验数一定为0
    [~, s] = min(Sigma);                                                   %选出最大检验数, 确定进基变量索引s；~表示忽略第一个参数（即最大值），s是索引
    Theta = b ./ A(:,s);                                                   %计算b/ai(点除）
    Theta(Theta<=0) = 10000;
    [~, q] = min(Theta);                                                   %选出最小θ
    q = index_Basis(q);                                                    %确定出基变量在系数矩阵中的列索引el, 主元为A(q,s)

%% 判断是否是最优解
    if ~any(Sigma < 0)                                                     %所有检验数都小于0，此基可行解为最优解, any表示存在某个检验数>0        
        xm = x0;
        fm = c' * xm;                                                      %算出最优解
        return
    end
%% 判断是否有解
    if all(A(:,s) <= 0)                                                    %表示检验数这一列每个数都<=0，有无界解
        xm = [];
        break
    end
%% 换基
    index_Basis(index_Basis == q) = s;                                     %新的基变量索引
    ind_Nonbasis = setdiff(1:n, index_Basis);                              %非基变量索引
%% 核心——旋转运算
    A(:,ind_Nonbasis) = A(:,index_Basis) \ A(:,ind_Nonbasis);              %核心——非基变量的部分等于（=）基变量索引的矩阵的逆乘剩余非基变量的矩阵
    b = A(:,index_Basis) \ b;                                              %核心——约束方程组常数项(=)基变量索引的矩阵的逆乘原约束方程常数项目
    A(:,index_Basis) = eye(m,m);                                           %核心——基变量索引的矩阵更换成单位矩阵
    noi=noi+1;
end
end
