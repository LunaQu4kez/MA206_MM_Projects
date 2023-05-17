import numpy as np
import pandas as pd

# 导入附件1
data = pd.read_excel(r"C:\Users\charl\Desktop\数学建模\Project\Project 4\附件1(Attachment 1)2023-51MCM-Problem B.xlsx").values

date = np.unique(data[:,0]) # 日期
city = np.unique(data[:,[1,2]]) # 城市

#选取“收货量”相关指标

t1 =[] # 收货总量
t2 =[] # 日均收货量
t3 =[] # 单日最大收货量
t4 =[] # 日收货量极差
for i in range(city.shape[0]):
    if data[data[:,2]==city[i]].shape[0]!=0:
        t1+= [np.sum(data[data[:,2]==city[i]][:,-1])]
        t2+= [np.mean(data[data[:,2]==city[i]][:,-1])]
        t3+= [np.max(data[data[:,2]==city[i]][:,-1])]
        t4+= [np.ptp(data[data[:,2]==city[i]][:,-1])]
    else:
        t1+= [0]
        t2+= [0]
        t3+= [0]
        t4+= [0]
t1 = np.array(t1)[:,None]
t2 = np.array(t2)[:,None]
t3 = np.array(t3)[:,None]
t4 = np.array(t4)[:,None]

#选取“发货量”相关指标
t5 =[] # 发货总量
t6 =[] # 日均发货量
t7 =[] # 单日最大发货量
t8 =[] # 日发货量极差        @
for i in range(city.shape[0]):
    if data[data[:,1]==city[i]].shape[0]!=0:
        t5+= [np.sum(data[data[:,1]==city[i]][:,-1])]
        t6+= [np.mean(data[data[:,1]==city[i]][:,-1])]
        t7+= [np.max(data[data[:,1]==city[i]][:,-1])]
        t8+= [np.ptp(data[data[:,1]==city[i]][:,-1])]
    else:
        t5+= [0]
        t6+= [0]
        t7+= [0]
        t8+= [0]
t5 = np.array(t5)[:,None]
t6 = np.array(t6)[:,None]
t7 = np.array(t7)[:,None]
t8 = np.array(t8)[:,None]

#选取"快递数量变化"相关指标
nd1 = np.zeros(shape=(date.shape[0],city.shape[0])) # 城市每日的发货
nd2 = np.zeros(shape=(date.shape[0],city.shape[0])) # 城市每日的收货
for i in range(date.shape[0]):
    d1 = data[data[:,0]==date[i]]
    for j in range(city.shape[0]):
        d2 = d1[d1[:,1]==city[j]] # 发
        d3 = d1[d1[:,2]==city[j]] # 收
        nd1[i,j] = np.sum(d2[:,-1])
        nd2[i,j] = np.sum(d3[:,-1])
nd11 = np.array([nd1[i+1]-nd1[i] for i in range(nd1.shape[0]-1)])
nd22 = np.array([nd2[i+1]-nd2[i] for i in range(nd2.shape[0]-1)])

t9 = nd11.max(axis=0)[:,None] # 发货量最大增幅
t10 = nd11.min(axis=0)[:,None] # 发货量最小增幅
t11 = nd11.mean(axis=0)[:,None] # 发货量平均增幅
t12 = nd11.std(axis=0)[:,None] # 发货量增幅标准差

t13 = nd22.max(axis=0)[:,None] # 收货量最大增幅
t14 = nd22.min(axis=0)[:,None] # 收货量最小增幅
t15 = nd22.mean(axis=0)[:,None] # 收货量平均增幅
t16 = nd22.std(axis=0)[:,None] # 收货量增幅标准差

#选取"相关性"相关指标
t17 = [] # 上游发货城市总数
t18 = [] # 下游发货城市总数
for i in range(city.shape[0]):
    d1 = data[data[:,2]==city[i]][:,1]
    t17+=[np.unique(d1).shape[0]]
    d2 = data[data[:,1]==city[i]][:,2]
    t18+=[np.unique(d2).shape[0]]

md1 =  np.zeros(shape=(date.shape[0],city.shape[0])) # 每日上游城市数
md2 =  np.zeros(shape=(date.shape[0],city.shape[0])) # 每日下游城市数
for i in range(date.shape[0]):
    d1 = data[data[:,0]==date[i]]
    for j in range(city.shape[0]):
        md1[i,j] = np.unique(d1[d1[:,2]==city[j]][:,1]).shape[0]
        md2[i,j] = np.unique(d1[d1[:,1]==city[j]][:,2]).shape[0]
t19 = md1.max(axis=0)[:,None] # 单日最大上游城市数
t20 = md2.max(axis=0)[:,None] # 单日最大下游城市数


datat = t1.copy()
for i in range(2,21):
    t = eval("t%d"%i)
    datat = np.c_["1",datat,t]


col = ["收货总量","日均收货量","单日最大收货量","日收货量极差","发货总量","日均发货量","单日最大发货量","日发货量极差",
"发货量最大增幅","发货量最小增幅","发货量平均增幅","发货量增幅标准差","收货量最大增幅","收货量最小增幅","收货量平均增幅",
"收货量增幅标准差","上游发货城市总数","下游发货城市总数","单日最大上游城市数","单日最大下游城市数"]

# 对数据进行正向化
t4 = (t4.max()-t4)/t4.ptp()
t8 = (t8.max()-t8)/t8.ptp()
t12 = (t12.max()-t12)/t12.ptp()
t16 = (t16.max()-t16)/t16.ptp()

datat1 = t1.copy()
for i in range(2,21):
    t = eval("t%d"%i)
    datat1 = np.c_["1",datat1,t]

# 对数据进行归一化
datat1 = (datat1-datat1.min(axis=0))/datat1.ptp()


# 使用熵权法加权
def EWM(A):
    """
    熵权法 the entropy weight method
    参数说明：
    A ：为原始数据矩阵A=a(i,j),表示第i个对象的j个指标,数据结构为np.array,shape=(n,m)
    返回值说明
    return ST,P,E,G,W,S
    ST : 得分排名从小到大
    P : 概率矩阵
    E : 指标的熵
    G : 指标混乱度
    W : 指标权重
    S : 评价对象得分
    """
    n,m = A.shape
    if 0 in A:
        A += 0.00001
    P = A/A.sum(axis=0)
    E = (-1/np.log(n))*np.sum(P*np.log(P),axis=0)
    G = 1-E
    W = G/G.sum()
    W = np.round(W,4)

    return W
w = EWM(datat1)

datat2 = datat1*w


def min_to_max(x):
    y = (np.max(x)-x)/(np.max(x)-np.min(x))
    return y.copy()
def mid_to_max(x,c):
    y = np.zeros_like(x)
    y[x>c] =1-(x[x>c]-c)/np.ptp(x)
    y[x<=c] =1-(c-x[x<=c])/np.ptp(x)
    return y.copy()

def range_to_max(x,a,b):
    y = np.ones_like(x)
    y[x>b] =1-(x[x>b]-b)/np.ptp(x)
    y[x<a] =1-(a-x[x<a])/np.ptp(x)
    return y.copy()
def TOPSIS(data,fm1=[],fm2=[],c=[],fm3=[],a=[],b=[],w=None):
    values = data.copy()
    if w == None:
        w = np.ones(shape=(1,values.shape[1]))
    for i in range(len(fm1)):
        values[:,fm1[i]] = min_to_max(values[:,fm1[i]])
    for i in range(len(fm2)):
        values[:,fm2[i]] = mid_to_max(values[:,fm2[i]],c[i])
    for i in range(len(fm3)):
        values[:,fm3[i]] = range_to_max(values[:,fm3[i]],a[i],b[i])
    values1 = (values-values.min(axis=0))/values.ptp(axis=0)
    values2 = w*values1
    M = values2.max(axis=0)
    m = values2.min(axis=0)
    D = np.sum((values2-M)**2,axis=1)**0.5
    d = np.sum((values2-m)**2,axis=1)**0.5
    f = d/(d+D)
    return f
TOPSIS(datat2)

