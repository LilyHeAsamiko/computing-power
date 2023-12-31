import numpy as np
IdxGlobal = [82,71,58,56,55,53,49,43,43,43,42,41,
43,40,36,30]
I0 = [86,75,53,56,55,53,49,43,43,43,42,41,43,40,36,30] 
W0 = [81,42,32]
I1 = [91,70,44,38,63,38,35,26,27,28,31,26,30,23,21]
W1 = [81,36,25]
I2 = [84,82,59,58,56,58,49,35,40,49,46,48,39,38,22]
W2 = [83,50,33]
I3 = [82,76,63,52,42,37,51,30,33,31,28,28,35,24,20]
W3 = [79,40,26]
I4 = [75,79,76,68,30,70,57,88,39,55,32,40,40,50,34]
W4 = [77,56,41]
I5 = [83,79,46,52,35,46,40,38,32,33,35,28,31,35,22]
W5 = [81,39,29] 

IdxR = [70,60,57,60,59,58,57,53,57,58,58,57,47,42,39]
WR = [65,57,43]
I0R = [57,55,59,60,59,58,57,53,57,58,58,57,
47,42,39] 
W0R = [65,57,43]
I1R = [57,55,59,60,63,55,55,54,55,56,55,50,53,48,45]
W1R = [56,56,49]
I2R = [68,61,65,68,63,65,65,61,58,61,62,58,54,45,45]
W2R = [65,61,41]
I3R = [61,59,52,63,55,55,51,50,49,60,62,53,49,34,36]
W3R = [60,55,40]
I4R = [70,59,65,66,66,61,60,50,60,64,60,62,53,38,33]
W4R = [65,61,41]
I5R = [84,63,58,56,61,56,61,58,67,57,59,62,36,45,36]
W5R = [74,60,39] 

IdxA = [82,72,70,68,64,65,61,56,62,58,55,53,43,34,30]
WA = [77,61,36]
I0A = [82,67,69,68,62,64,65,50,62,58,55,52,38,36,23] 
W0A = [75,61,32]
I1A = [82,57,76,75,65,76,64,60,63,50,50,58,39,27,27]
W1A = [77,61,36]
I2A = [80,84,76,71,71,66,65,65,60,60,60,51,47,39,35]
W2A = [82,65,40]
I3A = [81,77,67,67,66,59,54,55,58,57,57,55,39,25,25]
W3A = [79,60,30]
I4A = [72,80,64,60,55,58,55,42,57,65,58,51,35,24,23]
W4A = [76,57,27]

IdxB = [84,68,66,62,56,61,52,50,54,51,47,54,45,38,36]
WB = [76,55,40]
I0B = [85,74,68,64,56,64,52,56,59,52,46,64,43,42,34]
W0B = [80,58,40]
I1B = [85,65,70,69,68,69,67,59,68,68,69,68,63,53,52]
W1B = [77,61,36]
I2B = [82,41,49,48,48,50,46,40,45,40,40,42,31,29,25]
W2B = [62,45,28]
I3B = [82,80,69,71,59,60,50,45,50,55,45,43,36,33,30]
W3B = [81,55,33]
I4B = [82,80,70,64,56,60,52,55,49,45,41,53,47,35,35]
W4B = [81,55,39]

#5:5542
A0 = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
A1 = [[1,2,1.5,1,2],[1/2,1,1,1,1],[1/1.5,1,1,1,1],[1,1,1,1,1],[1/2,1,1,1,1]]
A2 = [[1,2,3,4,5],[1,1/2,1/3,1/4,1/5],[1,1/2,1/3,1,1],[1,1,1,1,1],[1,1,1,1,1]]
A3 = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
A4 = [[1,2],[1,2]]

#Y  Pearson相关系数
r_P = np.cov(I1, I2)/np.std(I1)/np.std(I2)
#[0.98707118,1.0625808,1.0625808,1.16299534]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示
plt.rcParams['axes.unicode_minus'] = False #解决符号无法显示
#导入
#data = pd.read_csv('data.csv')
#data.head()
from sklearn.preprocessing import StandardScaler # 创建标准化处理对象 
scaler = StandardScaler() # 对数据进行标准化处理 
scaled_data = scaler.fit_transform(np.array([I1,I2]))

from factor_analyzer import FactorAnalyzer, calculate_kmo,calculate_bartlett_sphericity
kmo = calculate_kmo(np.array([I1,I2]))
bartlett = calculate_bartlett_sphericity(np.array([I1,I2]))
print(f'KMO:{kmo[1]}')
print(f'Bartlett:{bartlett[1]}')

Load_Matrix = FactorAnalyzer(rotation=None, n_factors=len(scaled_data),method ='principal')
Load_Matrix.fit(scaled_data)
f_contribution_var = Load_Matrix.get_factor_variance()
matrices_var = pd.DataFrame(scaled_data)
matrices_var["旋转前特征值"] = f_contribution_var[0]
matrices_var["旋转前方差贡献率"] = f_contribution_var[1]
matrices_var["旋转前方差累计贡献率"] = f_contribution_var[2]
matrices_var

eigenvalues = 1
N = 0
for c in matrices_var["旋转前特征值"]:
    if c >= eigenvalues:
        N += 1
    else:
        s = matrices_var["旋转前方差累计贡献率"][N-1]
        print("\n选择了" + str(N) + "个因子累计贡献率为" + str(s)+"\n")
        break


# 主要用来看取多少因子合适，一般是取到平滑处左右，当然还要需要结合贡献率
import matplotlib
matplotlib.rcParams["font.family"] = "SimHei"  
ev, v = Load_Matrix.get_eigenvalues()
print('\n相关矩阵特征值：', ev)
plt.figure(figsize=(8, 6.5))
plt.scatter(range(1, scaled_data.shape[1] + 1), ev)
plt.plot(range(1, scaled_data.shape[1] + 1), ev)
plt.title('特征值和因子个数的变化', fontdict={'weight': 'normal', 'size': 25})
plt.xlabel('因子', fontdict={'weight': 'normal', 'size': 15})
plt.ylabel('特征值', fontdict={'weight': 'normal', 'size': 15})
plt.grid()
plt.show()  

Load_Matrix_rotated = FactorAnalyzer(rotation='varimax', n_factors=2, method='principal')
Load_Matrix_rotated.fit(scaled_data)
f_contribution_var_rotated = Load_Matrix_rotated.get_factor_variance()
matrices_var_rotated = pd.DataFrame(scaled_data)
matrices_var_rotated["特征值"] = f_contribution_var_rotated[0]
matrices_var_rotated["方差贡献率"] = f_contribution_var_rotated[1]
matrices_var_rotated["方差累计贡献率"] = f_contribution_var_rotated[2]
print("旋转后的载荷矩阵的贡献率")
print(matrices_var_rotated)
print("旋转后的成分矩阵")
print(Load_Matrix_rotated.loadings_)

import seaborn as sns
import numpy as np
Load_Matrix = Load_Matrix_rotated.loadings_
#df = pd.DataFrame(np.abs(Load_Matrix),index= scaled_data.columns)
df = pd.DataFrame(np.abs(Load_Matrix))
 
plt.figure(figsize=(8, 8))
ax = sns.heatmap(df, annot=True, cmap="BuPu",cbar=False)
ax.yaxis.set_tick_params(labelsize=15) # 设置y轴字体大小
plt.title("因子分析", fontsize="xx-large")
plt.ylabel("因子", fontsize="xx-large")# 设置y轴标签
plt.show()# 显示图片

'''# 计算因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
f_corr = (scaled_data).corr()  # 皮尔逊相关系数
X1 = np.mat(f_corr)
X1 = np.linalg.inv(X1)
factor_score_weight = np.dot(X1, Load_Matrix_rotated.loadings_)
factor_score_weight = pd.DataFrame(factor_score_weight)
col = []
for i in range(N):
    col.append("factor" + str(i + 1))
factor_score_weight.columns = col
factor_score_weight.index = f_corr.columns
print("因子得分：\n", factor_score_weight)'''

import numpy as np
from scipy import stats
α,A=0.05,np.array([I1,I2])
if A.sum()>=50:
    print('X/Y',end='\t')
    for i in range(A.shape[0]):print('Y'+str(i+1),end='\t')
    print('合计')
    for i in range(A.shape[0]):
        print('X'+str(i+1),end='\t')
        for j in range(A.shape[1]):print(A[i,j],end='\t')
        print(sum(A[i,:]))
    print('合计',end='\t')
    for j in range(A.shape[1]):print(sum(A[:,j]),end='\t')
    print(A.sum());print('-'*A.shape[1]*8)
    tongjiliang=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            tongjiliang+=(A.sum()*A[i,j]-sum(A[i,:])*
            sum(A[:,j]))**2/(A.sum()*sum(A[i,:])*sum(A[:,j]))
    m=(A.shape[0]-1)*(A.shape[1]-1)
    print('X²='+str('%.3f'%tongjiliang))
    print('X²('+str(α)+','+str(m)+')='+str('%.3f'%stats.chi2.isf(α,m)))
    if tongjiliang>stats.chi2.isf(α,m):
        print('X²>X²('+str(α)+','+str(m)+')')
        print('两因素不独立')
    else:
        print('X²<=X²('+str(α)+','+str(m)+')')
        print('两因素独立')
    print('-'*A.shape[1]*8)
else:print('样本量小于50，不能进行双因素独立性检验！')

X/Y     Y1      Y2      合计
X1      91      70      44      38      63      38      35      26      27      28      31      26      30      23      21       591
X2      84      82      59      58      56      58      49      35      40      49      46      48      39      38      22       763
合计    175     152     103     96      119     96      84      61      67      77      77      74      69      61      43       1354
------------------------------------------------------------------------------------------------------------------------
X²=16.838
X²(0.05,14)=23.685
X²<=X²(0.05,14)
两因素独立
------------------------------------------------------------------------------------------------------------------------

G = [58000, 61000, 62000, 60000, 68000, 69000, 71000]
D = [25000, 27000, 28000, 29000, 30000, 32000, 35000]

class lasso_regression():
    def _init_(self,learning_rate= 0.01, n = 1000):
        self.learning_rate = learning_rate
        self.n = n
        self.L1 = L1

    def coordinate(self):
        for i in range(self.num + 1):
            a = (self.X[:,i].T.dot(*self.X[:,i].reshape(1,-1)))
            dw = np.matrix(np.zeros((1,self.num +1)))
            dw[0,i] = self.w[i]
            b = 2*(self.X.T*(self.Y.reshape(-1,1)-self.X.dot((self.w-dw).T)))[i,0]
            if b <-self.L1:
                self.w[i] =(-self.L1-b)/a/2
            elif b > slef.L1:
                self.w[i] = (self.L1-b)/a/2
            else:
                self.w[i] = 0
        self.w = self.w - self.gradient()*slef.learning_rate

    def costfun(self):
        return 1/(2*self.m)*((self.X.dot(self.w.reshape(-1,1))-self.Y.reshape(-1,1))**2).sum() + self.L1*(abs(self.w)).sum()

    def fit(self,X,Y):
        self.w = np.array([0 for x in range(X.shape[1]+1)])
        self.m = X.shape[0]
        self.X = np.c_[X,[1 for x in range(self.m)]]
        self.Y = Y
        cost1 = self.costfun()
        cost2 = 0
        for i in range(self.n):
            if cost1 - cost2 <= 1e-6
                break
            else:
                cost1 = self.costfun()
                self.renew()
                cost2 = self.costfun()

    def predict(self,xt):
        xt = np.c_[xt, [1 for x in range(xt.shape[0])]]
        return xt.dot(slef.w.T)

class linear_regression():
    def _init_(self,learning_rate= 0.01, n = 1000):
        self.learning_rate = learning_rate
        self.n = n

    def gradient(self):
        return ((1/self.m*(self.X.dot(self.w.reshape(-1,1))-self.Y.reshape(-1,1))).reshape(1,-1).dot(self.X)
    
    def renew(self):
        self.w = self.w - self.gradient()*slef.learning_rate

    def costfun(self):
        return 1/(2*self.m)*((self.X.dot(self.w.reshape(-1,1))-self.Y.reshape(-1,1))**2).sum()

    def fit(self,X,Y):
        self.w = np.array([0 for x in range(X.shape[1]+1)])
        self.m = X.shape[0]
        self.X = np.c_[X,[1 for x in range(self.m)]]
        self.Y = Y
        cost1 = self.costfun()
        cost2 = 0
        for i in range(self.n):
            if cost1 - cost2 <= 1e-6
                break
            else:
                cost1 = self.costfun()
                self.renew()
                cost2 = self.costfun()

    def predict(self,xt):
        xt = np.c_[xt, [1 for x in range(xt.shape[0])]]
        return xt.dot(slef.w.T)
    
   
matrices_var
   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
0  91  70  44  38  63  38  35  26  27  28  31  26  30  23  21
1  84  82  59  58  56  58  49  35  40  49  46  48  39  38  22

scaled_data
array([[-1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.],
       [ 1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.]])

 G = [58000, 61000, 62000, 60000, 68000, 69000, 71000]
D = [25000, 27000, 28000, 29000, 30000, 32000, 35000]        
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,LLLLLL,test_size=0.3,random_state=420)

reg = LR().fit(xtrain,ytrain)
yhat = reg.predict(xtest)
coef = reg.coef_

intercept = reg.intercept_
intercept
0.04462145053255845

from sklearn.metrics import mean_squared_error as MSE
mse = MSE(yhat,ytest)
mse
0.08708507063702606

cross = cross_val_score(reg,x,y,cv=10,scoring="neg_mean_squared_error")

from sklearn.metrics import r2_score
r_score = r2_score(y_true=ytest,y_pred=yhat)
r_score
-0.2925283735394537
score = reg.score(xtest,ytest)
score
-0.2925283735394537

sorted(ytest)
[0.2340516625912965, 0.463699789295839, 0.7985494179892573, 0.8301732157072035, 0.9231824966275339]
plt.plot(range(len(ytest)),sorted(ytest),c='black',label='Data')

plt.plot(range(len(yhat)),sorted(yhat),c='red',label='Predict')
plt.legend()
plt.show()

X.shift(1)

plt.plot(range(len(ytestD)),sorted(ytestD),c='black',label='Data')

plt.plot(range(len(yhat)),sorted(yhat),c='red',label='Predict')
plt.legend()
plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['2016','2017', '2018', '2019', '2020', '2021','2022','2023']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, D, width, label='数字经济')
rects2 = ax.bar(x + width/2, G, width, label='GDP')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('USD billion')
ax.set_title('GDP and Digital Economics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()