# -*-coding:utf-8-*-

import itertools
import os
from 实时接口.Ashare import *
from 工具.MyTT import *
import numpy as np
import baostock as bs
import pandas as pd
import time
import warnings
import feather
import itertools as it
from scipy import stats
import math
import pandas as pd
from tqdm import tqdm
# <editor-fold desc="框架工具函数">
def 合成k线求high(a, barCount):
    barCount = int(barCount)
    data = pd.DataFrame()
    data["a"] = a
    max = pd.Series(data['a'].rolling(barCount).max())
    return max.values
def 合成k线求low(a, barCount):
    barCount = int(barCount)
    data = pd.DataFrame()
    data["a"] = a
    min = pd.Series(data['a'].rolling(barCount).min())
    return min.values
def 合成k线求合成量能(a, barCount):
    barCount = int(barCount)
    data = pd.DataFrame()
    data["a"] = a
    sum = pd.Series(data['a'].rolling(barCount).sum())
    return sum.values


def barslast(输入):
    输入 = np.array(输入)
    输入 = (~(输入.astype(bool))).astype(int)
    # 得到连续的1有多少个
    # 做一次前缀和
    mask = np.insert(输入, 0, 0)
    sum_m = np.cumsum(mask)
    # 取出0的位置的值
    sum_0 = sum_m[mask == 0]
    # 这些位置的值去重相减，就是每个连续的1的数量
    continue1 = np.diff(np.unique(sum_0))
    # 把1变成0的位置的值变成连续的1的数量的负数
    temp = 输入.copy()
    position = np.where(np.diff(输入) == -1)[0] + 1
    temp[position] = -continue1
    # 求前缀和，从1变成0时，会自动加上连续的1的个数的负数，则自动正确
    多方持续时间 = np.cumsum(temp)
    return 多方持续时间

def FILTER(a,n):
    return a
def CEILING(a):
    return a



def hhv(a, n):
    try:
        test = pd.DataFrame()
        test["a"] = a
        ans = test["a"].rolling(n).max().values
        return ans
    except:

        try:
            a.tolist()
        except:
            1
        try:
            n.tolist()
        except:
            1
        return np.array(maxBarslast(a, n))


def llv(a, n):
    try:
        n=int(n)
        test = pd.DataFrame()
        test["a"] = a
        ans = test["a"].rolling(n).min().values
        return ans
    except:

        try:
            a.tolist()
        except:
            1
        try:
            n.tolist()
        except:
            1
        return np.array(minBarslast(a,n))






# def sma(S, N, M=1):  # 中国式的SMA,至少需要120周期才精确
#     K = pd.Series(S).rolling(N).mean()  # 先求出平均值
#     for i in range(N + 1, len(S)):  K[i] = (M * S[i] + (N - M) * K[
#         i - 1]) / N  # 因为要取K[i-1]，所以 range(N+1, len(S))
#     return K
def sma(S, N, M):  # 3）高效写法3
    try:
        return pd.Series(S).ewm(span=2 * N / M - 1, adjust=True).mean().values
    except:
        try:
            try:
                M.tolist()
            except:
                1
            return SMA_M变动(S, N, M)
        except:
            try:
                try:
                    N.tolist()
                except:
                    1
                return SMA_N变动(S, N, M)
            except:
                try:
                    try:
                        M.tolist()
                    except:
                        1
                    try:
                        N.tolist()
                    except:
                        1
                    return SMA_双变动(S, N, M)
                except:
                    1


from numba import jit, njit
@jit(nopython=True)
def SMA_双变动(arr, n_values, m_values):
    result = []
    y = 0
    n_index = 0
    m_index = 0

    for x in arr:
        if x != x:  # Check if x is NaN (NaN values are not equal to themselves)
            x = 0  # Replace NaN with 0

        n = int(n_values[n_index])  # Get the corresponding n value for the current data point
        n_index += 1  # Move to the next index for n_values

        m = int(m_values[m_index])  # Get the corresponding m value for the current data point
        m_index += 1  # Move to the next index for m_values

        y = (m * x + (n - m) * y) / n
        result.append(y)

    return result


@jit(nopython=True)
def SMA_N变动(arr, n, m_values):
    m = int(m_values)
    y = 0
    result = []
    for x, n in zip(arr, n):
        if isinstance(x, float) and np.isnan(x):  # 判断是否为 NaN 值
            x = 0  # 替换 NaN 值为 0
        y = (m * x + (n - m) * y) / n
        result.append(y)
    return result


@jit(nopython=True)
def SMA_M变动(arr, n, m_values):
    n = int(n)
    y = 0
    result = []
    m_index = 0  # Initialize an index to track the current m value
    for x in arr:
        m = m_values[m_index] if m_index < len(m_values) else m_values[-1]  # Use last value if not enough values in m_values
        if np.isnan(x):
            x = np.nan_to_num(x)
        y = (m * x + (n - m) * y) / n
        result.append(y)
        m_index += 1  # Move to the next m value
    return np.array(result)

def SMA_CN(arr, n, m):
    n = int(n)
    m = int(m)
    y = 0
    result = []
    for x in arr:
        if np.isnan(x):
            x = np.nan_to_num(x)
        y = (m * x + (n - m) * y) / n
        result.append(y)
    return np.array(result)




def EMA(S, N):  # 为了精度 S>4*N  EMA至少需要120周期
    try:
        return pd.Series(S).ewm(span=N, adjust=False).mean().values
    except:
        try:
            S.tolist()
        except:
            1
        try:
            N.tolist()
        except:
            1
        S = np.array(S).astype(float)
        N = np.array(N).astype(int)
        return EMA_变动(S, N)

        return EMA_变动(S,N)


@jit(nopython=True)
def EMA_变动(S, N):

    ema_values = []
    ema = S[0]  # Initialize the first EMA as the first value in the sequence
    ema_values.append(0)
    for i in range(1, len(S)):
        alpha = 2 / (N[i - 1] + 1)
        ema = alpha * S[i] + (1 - alpha) * ema
        ema_values.append(ema)

    return ema_values


@jit(nopython=True)
def countBarslast(a, n):
    ans = []
    for i in range(len(a)):
        window = n[i] - 1
        calc = sum(a[i - window:i + 1])
        ans.append(calc)
    return ans
@jit(nopython=True)
def maxBarslast(a, n):
    ans = []
    for i in range(len(a)):
        window = n[i] - 1
        try:
            calc = max(a[i - window:i + 1])
            ans.append(calc)
        except:
            ans.append(0)
    return ans

@jit(nopython=True)
def minBarslast(a, n):
    ans = []
    for i in range(len(a)):
        window = n[i] - 1
        try:
            calc = min(a[i - window:i + 1])
            ans.append(calc)
        except:
            ans.append(0)
    return ans

def REF(S, N=1):       #对序列整体下移动N,返回序列(shift后会产生NAN)
    try:
        return pd.Series(S).shift(N).values

    except:
        try:
            S.tolist()
        except:
            1
        try:
            N.tolist()
        except:
            1
        return np.array(refBarslast(S, N))


def MA(S,N):           #求序列的N日平均值，返回序列
    try:
        N=int(N)
        return pd.Series(S).rolling(N).mean().values
    except:
        try:
            S.tolist()
        except:
            1
        try:
            N.tolist()
        except:
            1

        return dynamic_MA(S,N)

@jit(nopython=True)
def dynamic_MA(S, N):
    results = []
    for i in range(len(S)):
        window_size = int(N[i])
        total = 0
        count = 0
        for j in range(i, max(-1, i - window_size), -1):
            total += S[j]
            count += 1
        result = total / count
        results.append(result)
    return results


def DMA(X, A):
        try:
            if A<1:
                return DMA_计算(X,A)
            else:
                return np.full(len(X),0)
        except:
            return np.full(len(X),0)


@jit(nopython=True)
def DMA_计算(X, A):
    if A < 1:
        Y = [0.0 for i in range(len(X))]   # 初始化动态移动平均结果列表
        Y[0] = X[0]  # 将第一个元素作为初始值
        for i in range(1, len(X)):
            Y[i] = A * X[i] + (1 - A) * Y[i - 1]
        return Y



def CROSS(S1,S2):                      #判断穿越 CROSS(MA(C,5),MA(C,10))
    try:
        len(S2)
        ans=np.multiply(S1>S2,REF(S1,1)<REF(S2,2))
        return ans    #上穿：昨天0 今天1   下穿：昨天1 今天0
    except:
        S2=np.full(len(S1),S2)
        ans=np.multiply(S1>S2,REF(S1,1)<REF(S2,2))
        return ans    #上穿：昨天0 今天1   下穿：昨天1 今天0
@jit(nopython=True)
def refBarslast(a, n):
    ans = []
    for i in range(len(a)):
        window = n[i]
        calc = a[i - window]
        ans.append(calc)
    return ans
@jit(nopython=True)
def yijin(a):
    result = []
    for i in range(len(a)):
        if i >= 5:
            window = a[i - 5:i + 1]
            number = window[0] * 32 + window[1] * 16 + window[2] * 8 + window[3] * 4 + window[4] * 2 + window[5] * 1
        else:
            number = -1
        result.append(number)
    return result
@jit(nopython=True)
def standard_yijin(a):
    result_ben = []
    result_bian = []
    result_cuo = []
    result_zong = []
    result_hu = []
    result_dongyao = []
    for i in range(len(a)):
        if i >= 17:
            window = a[i - 17:i + 1]
            x1 = window[0] + window[1] + window[2]
            x2 = window[3] + window[4] + window[5]
            x3 = window[6] + window[7] + window[8]
            x4 = window[9] + window[10] + window[11]
            x5 = window[12] + window[13] + window[14]
            x6 = window[15] + window[16] + window[17]

            ben = (x6 >= 2) * 32 + (x5 >= 2) * 16 + (x4 >= 2) * 8 + (x3 >= 2) * 4 + (x2 >= 2) * 2 + (x1 >= 2) * 1
            bian = (x6 == 0 or x6 == 2) * 32 + (x5 == 0 or x5 == 2) * 16 + (x4 == 0 or x4 == 2) * 8 + (
                    x3 == 0 or x3 == 2) * 4 + (x2 == 0 or x2 == 2) * 2 + (x1 == 0 or x1 == 2) * 1
            cuo = (x6 < 2) * 32 + (x5 < 2) * 16 + (x4 < 2) * 8 + (x3 < 2) * 4 + (x2 < 2) * 2 + (x1 < 2) * 1
            zong = (x1 >= 2) * 32 + (x2 >= 2) * 16 + (x3 >= 2) * 8 + (x4 >= 2) * 4 + (x5 >= 2) * 2 + (x6 >= 2) * 1
            hu = (x5 >= 2) * 32 + (x4 >= 2) * 16 + (x3 >= 2) * 8 + (x4 >= 2) * 4 + (x3 >= 2) * 2 + (x2 >= 2) * 1
            dongyao = (x6 == 0 or x6 == 3) * 32 + (x5 == 0 or x5 == 3) * 16 + (x4 == 0 or x4 == 3) * 8 + (
                    x3 == 0 or x3 == 3) * 4 + (x2 == 0 or x2 == 3) * 2 + (x1 == 0 or x1 == 3) * 1
        else:
            ben = -1
            bian = -1
            cuo = -1
            zong = -1
            hu = -1
            dongyao = -1
        result_ben.append(ben)
        result_bian.append(bian)
        result_hu.append(hu)
        result_cuo.append(cuo)
        result_zong.append(zong)
        result_dongyao.append(dongyao)
        final = [result_ben, result_bian, result_hu, result_cuo, result_zong, result_dongyao]

    return final
@jit(nopython=True)
def pinghua_yijin(a, b):
    result_ben = []
    result_bian = []
    result_cuo = []
    result_zong = []
    for i in range(len(a)):
        if i >= 5:
            window_a = a[i - 5:i + 1]
            window_b = b[i - 5:i + 1]

            ben = (window_a[5] == 1 and window_b[5] == 1) * 32 + (window_a[4] == 1 and window_b[4] == 1) * 16 + (
                    window_a[3] == 1 and window_b[3] == 1) * 8 + (window_a[2] == 1 and window_b[2] == 1) * 4 + (
                          window_a[1] == 1 and window_b[1] == 1) * 2 + (window_a[0] == 1 and window_b[0] == 1) * 1
            bian = (window_a[5] == 0 and window_b[5] == 0) * 32 + (window_a[4] == 0 and window_b[4] == 0) * 16 + (
                    window_a[3] == 0 and window_b[3] == 0) * 8 + (window_a[2] == 0 and window_b[2] == 0) * 4 + (
                           window_a[1] == 0 and window_b[1] == 0) * 2 + (window_a[0] == 0 and window_b[0] == 0) * 1
            cuo = (window_a[5] == 1 and window_b[5] == 0) * 32 + (window_a[4] == 1 and window_b[4] == 0) * 16 + (
                    window_a[3] == 1 and window_b[3] == 0) * 8 + (window_a[2] == 1 and window_b[2] == 0) * 4 + (
                          window_a[1] == 1 and window_b[1] == 0) * 2 + (window_a[0] == 1 and window_b[0] == 0) * 1
            zong = (window_a[5] == 0 and window_b[5] == 1) * 32 + (window_a[4] == 0 and window_b[4] == 1) * 16 + (
                    window_a[3] == 0 and window_b[3] == 1) * 8 + (window_a[2] == 0 and window_b[2] == 1) * 4 + (
                           window_a[1] == 0 and window_b[1] == 1) * 2 + (window_a[0] == 0 and window_b[0] == 1) * 1

        else:
            ben = -1
            bian = -1
            cuo = -1
            zong = -1

        result_ben.append(ben)
        result_bian.append(bian)
        result_cuo.append(cuo)
        result_zong.append(zong)

        final = [result_ben, result_bian, result_cuo, result_zong]

    return final
@jit(nopython=True)
def yijindaoshu(a, b, c, d, e):
    result = []
    # 本函数求易经如果全是1卦象就求强度的导数的卦象加上63
    for i in range(len(a)):
        if a[i] < 63:
            result.append(a[i])
        elif a[i] >= 63 and b[i] < 63:
            result.append(b[i] + 63)
        elif a[i] >= 63 and b[i] >= 63 and c[i] < 63:
            result.append(c[i] + 63 + 63)
        elif a[i] >= 63 and b[i] >= 63 and c[i] >= 63 and d[i] < 63:
            result.append(d[i] + 63 + 63 + 63)
        elif a[i] >= 63 and b[i] >= 63 and c[i] >= 63 and d[i] >= 63:
            result.append(e[i] + 63 + 63 + 63 + 63)
    return result
def count(输入, n):
    try:
        n = int(n)
        test = pd.DataFrame()
        test["输入"] = 输入
        counts = pd.Series(test['输入'].rolling(n).sum()).values
        return counts
    except:
        输入=np.array(输入).astype(float)
        n=np.array(输入).astype(int)
        return countBarslast(输入,n)

def BARSCOUNT(n):
    return np.full(len(n),9999)

def BARScount(n):
    return np.full(len(n),9999)




def 时间转换(timestamps):
    result = []
    for timestamp in timestamps:
        timestamp_str = str(int(timestamp))  # 转换为整型并转换为字符串
        year = timestamp_str[:4] + "年"
        month = timestamp_str[4:6] + "月"
        day = timestamp_str[6:8] + "日"
        hour = timestamp_str[8:10] + "点"
        minute = timestamp_str[10:12]
        result.append(year + month + day + hour + minute)
    return result
# </editor-fold>
# <editor-fold desc="print打印显示设置">
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
np.set_printoptions(formatter={'all': lambda x: str(x)}, threshold=100)

# </editor-fold>

from k线合成器.k线合成器 import k线合成器5min

def 策略(合成周期,ktime,high,o,low,c,volume,amount,均线,开盘,收盘,切片位置,subindex,turn):


    # =========兼容性写法=============================================================================
    HIGH = high
    OPEN = o
    LOW = low
    CLOSE = c
    VOL=volume
    V=volume
    H=high
    L=low
    O=o
    C=CLOSE
    AMO=amount
    HSL=turn
    padding = []
    转换关系=int(240/int(合成周期))
    # 迭代日线信号数组
    for daily_signal_value in HSL:
        # 重复日线信号60次以填充60分钟线信号
        padding.extend([daily_signal_value] * 转换关系)
    HSL=np.array(padding)
    AMOUNT=amount
    # ==============================开始编写特征============================================================es
    a=volume<REF(volume)
#开始



    均价=((2*C+(C+H+L+O)/4+O)/4)
    R=(3*L+2*均价)/5
    R1=(3*H+2*均价)/5
    KX=((R+R1)/2)
    跟庄线=(EMA(count(KX*V,4)/count(V,4),4))
    RD=((R1+H)/1.56)
    寻庄至尊1=(55*RD+54*REF(RD,1)+53*REF(RD,2)+52*REF(RD,3)+51*REF(RD,4)+50*REF(RD,5)+49*REF(RD,6)+48*REF(RD,7)+47*REF(RD,8)+46*REF(RD,9))
    寻庄至尊2=(45*REF(RD,10)+44*REF(RD,11)+43*REF(RD,12)+42*REF(RD,13)+41*REF(RD,14)+40*REF(RD,15)+39*REF(RD,16)+38*REF(RD,17)+37*REF(RD,18)+36*REF(RD,19))
    寻庄至尊3=(35*REF(RD,20)+34*REF(RD,21)+33*REF(RD,22)+32*REF(RD,23)+31*REF(RD,24)+30*REF(RD,25)+29*REF(RD,26)+28*REF(RD,27)+27*REF(RD,28)+26*REF(RD,29))
    寻庄至尊4=(25*REF(RD,30)+24*REF(RD,31)+23*REF(RD,32)+22*REF(RD,33)+21*REF(RD,34)+20*REF(RD,35)+19*REF(RD,36)+18*REF(RD,37)+17*REF(RD,38)+16*REF(RD,39))
    寻庄至尊5=(15*REF(RD,40)+14*REF(RD,41)+13*REF(RD,42)+12*REF(RD,43)+11*REF(RD,44)+10*REF(RD,45)+9*REF(RD,46)+8*REF(RD,47)+7*REF(RD,48)+6*REF(RD,49))
    寻庄至尊6=(5*REF(RD,50)+4*REF(RD,51)+3*REF(RD,52)+2*REF(RD,53)+1*REF(RD,54))
    寻庄至尊7=(寻庄至尊1+寻庄至尊2+寻庄至尊3+寻庄至尊4+寻庄至尊5+寻庄至尊6)
    寻庄至尊=(寻庄至尊7/1765)
    寻庄看底1=(55*R+54*REF(R,1)+53*REF(R,2)+52*REF(R,3)+51*REF(R,4)+50*REF(R,5)+49*REF(R,6)+48*REF(R,7)+47*REF(R,8)+46*REF(R,9))
    寻庄看底2=(45*REF(R,10)+44*REF(R,11)+43*REF(R,12)+42*REF(R,13)+41*REF(R,14)+40*REF(R,15)+39*REF(R,16)+38*REF(R,17)+37*REF(R,18)+36*REF(R,19))
    寻庄看底3=(35*REF(R,20)+34*REF(R,21)+33*REF(R,22)+32*REF(R,23)+31*REF(R,24)+30*REF(R,25)+29*REF(R,26)+28*REF(R,27)+27*REF(R,28)+26*REF(R,29))
    寻庄看底4=(25*REF(R,30)+24*REF(R,31)+23*REF(R,32)+22*REF(R,33)+21*REF(R,34)+20*REF(R,35)+19*REF(R,36)+18*REF(R,37)+17*REF(R,38)+16*REF(R,39))
    寻庄看底5=(15*REF(R,40)+14*REF(R,41)+13*REF(R,42)+12*REF(R,43)+11*REF(R,44)+10*REF(R,45)+9*REF(R,46)+8*REF(R,47)+7*REF(R,48)+6*REF(R,49))
    寻庄看底6=(5*REF(R,50)+4*REF(R,51)+3*REF(R,52)+2*REF(R,53)+1*REF(R,54))
    寻庄看底7=(寻庄看底1+寻庄看底2+寻庄看底3+寻庄看底4+寻庄看底5+寻庄看底6)
    寻庄看底=(寻庄看底7/1765)
    寻庄黄金=((寻庄看底+寻庄至尊)/2)
    DD=(np.abs(跟庄线-MA(寻庄黄金,60))/MA(寻庄黄金,60))
    黄龙=(EMA(寻庄黄金,60))
    遁地=(黄龙*(1-14/100))
    反转=((KX))
    LC =( REF(KX,1))
    RS1=(sma(np.maximum(KX-LC,0),11,1))
    RS2=(sma(np.abs(KX-LC),11,1))
    天使=((RS1*8*4)-(RS2*8))
    唇吻=(EMA((RS2*8/4)-(RS1*8),3))
    a=(CROSS(天使,唇吻))
#结束
    买入 = a
    #确定买入和测试信号------------------------------------------------------
    测试信号 =买入
    买入信号 = np.where(买入,1,0)



    if 合成周期==240:

        测试信号 = 测试信号
        买入信号 = 买入信号
        if __name__ == "__main__":
            # =================测试模块========================
            当周期全特征 = pd.DataFrame()
            当周期全特征["买入信号"] = 买入信号
            # =================测试指标========================
            当周期全特征["测试信号"] = 测试信号

            # ================================================
            当周期全特征["code"] = 测试代码
            当周期全特征["time"] = 时间转换(ktime)
            return 买入信号, 当周期全特征
        else:
            return np.divide(1,np.add(barslast(买入信号),1))

    if 合成周期 != 240:
        # 通用全级别特征初始化
        收盘 = subindex.astype(int) % 48 == 0
        开盘 = np.append(np.array([0.0]), 收盘)[:-1]

        买入信号=count(买入信号,barslast(开盘)+1)
        测试信号=count(测试信号,barslast(开盘)+1)
        还原位置 = 切片位置
        当周期全特征 = pd.DataFrame()

        当周期全特征["买入信号" ]=买入信号
        当周期全特征["subindex"] = subindex.astype(int)
        当周期全特征["quyu"] = 当周期全特征["subindex"] % 48
        日表 = 当周期全特征[当周期全特征["quyu"] == 0]
        买入信号=日表["买入信号"].values

        if __name__ == "__main__":
            当周期全特征["time"] = 时间转换(ktime)
            当周期全特征["code"] = 测试代码
            当周期全特征["测试信号"] = 买入信号
            return 买入信号, 日表
        else:
            return np.divide(1,np.add(barslast(买入信号),1))




if __name__=='__main__':



    测试代码="000006"
    合成周期=240
    个股全日 = feather.read_dataframe("../Data_module/5minData/" + 测试代码)

    合成k线 = k线合成器5min(测试代码,个股全日, 合成周期)
    ktime = 合成k线[0]
    high = 合成k线[1]
    o = 合成k线[2]
    low = 合成k线[3]
    c = 合成k线[4]
    volume = 合成k线[5]
    amount = 合成k线[6]
    均线 = 合成k线[7]
    收盘 = 合成k线[8]
    开盘 = 合成k线[9]
    切片位置 = 合成k线[10]
    subindex = 合成k线[11]
    turn=合成k线[12]
    isST=合成k线[13]

    买入信号,当周期全特征=策略(合成周期,ktime,high,o,low,c,volume,amount,均线,开盘,收盘,切片位置,subindex,turn)

    print(当周期全特征)