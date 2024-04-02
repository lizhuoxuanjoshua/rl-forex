import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def 标记新高点(data):
    新高点 = []
    当前最高 = data[0]
    for i, value in enumerate(data):
        if value > 当前最高:
            新高点.append((i, value))  # 记录新高点的位置和值
            当前最高 = value
    return 新高点

@jit(nopython=True)
def 生成虚拟新高点(data, 增长率=0.1, 虚拟新高数量=5):
    新高点 = 标记新高点(data)
    最后一个新高值 = 新高点[-1][1] if 新高点 else data[-1]
    虚拟新高点 = [最后一个新高值 * ((1 + 增长率) ** (i + 1)) for i in range(虚拟新高数量)]
    return 虚拟新高点


@jit(nopython=True)
def 平滑下降和虚拟新高生成(data, 虚拟新高数量, 增长率):
    # 生成虚拟新高点
    虚拟新高点 = 生成虚拟新高点(data, 增长率, 虚拟新高数量)
    # 计算平滑下降段
    平滑下降段 = [虚拟新高点[-1] * (1 - 增长率) ** i for i in range(1, 虚拟新高数量 + 1)]
    平滑下降段[-1] = data[-1]  # 确保平滑下降段结束于原始数据最右点的值

    # 合并原始数据（除最后一点避免重复）、虚拟新高点和平滑下降段
    结果数据 = np.append(np.append(data[:-1],虚拟新高点), 平滑下降段)

    return 结果数据
@jit(nopython=True)
def 判断并生成虚拟新高点(data, 增长率=0.05, 虚拟新高数量=5):
    # 首先标记新高点
    新高点 = 标记新高点(data)

    # 如果最后一个数据点等于最后一个新高点的值，则认为创新高
    if 新高点 and data[-1] == 新高点[-1][1]:
        # 创新高，生成虚拟新高点
        虚拟新高点 = 平滑下降和虚拟新高生成(data,虚拟新高数量, 增长率 )

        return 虚拟新高点
    else:
        # 没有创新高，不生成虚拟新高点
        return data



def 绘制新高和虚拟新高(data, 虚拟新高点):
    新高点 = 标记新高点(data)
    x_data = list(range(len(data)))

    if 新高点:
        新高点位置, 新高点值 = zip(*新高点)
    else:
        新高点位置, 新高点值 = [], []

    虚拟新高点位置 = [len(data) + i for i in range(1, len(虚拟新高点) + 1)]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, data, label='Original Data', marker='o', color='gray')
    plt.scatter(新高点位置, 新高点值, color='red', label='New Highs')
    plt.scatter(虚拟新高点位置, 虚拟新高点, color='blue', label='Virtual New Highs')

    plt.legend()
    plt.title('New Highs and Virtual New Highs')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.show()

if __name__ == '__main__':

    # 示例数据
    data = [2, 3, 1, 5, 4, 6]

    # 计算虚拟新高点
    虚拟新高点 = 判断并生成虚拟新高点(data)

    # 绘制新高和虚拟新高
    绘制新高和虚拟新高(data, 虚拟新高点)
