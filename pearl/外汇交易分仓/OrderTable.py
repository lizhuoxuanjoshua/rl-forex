import numpy as np
import pandas as pd
import os
import feather

# 文件路径
data_file = 'data_table.feather'

# 初始化 DataFrame 的列，包括一个唯一的ID列
columns = ['ID', '多单', '空单','下单价','摩擦成本', '止损价','市值','开仓市值','最大市值','最小市值']

# 检查文件是否存在来决定是否从文件读取数据或初始化空的 DataFrame
if os.path.exists(data_file):
    # 从文件读取数据
    data = feather.read_dataframe(data_file)
else:
    # 创建空的 DataFrame 并初始化 ID 计数器
    data = pd.DataFrame(columns=columns)
    data['ID'] = pd.Series(dtype='int')

# 确保ID列是唯一且递增的
if 'ID' not in data.columns or data['ID'].empty:
    data['ID'] = range(1, len(data) + 1)

current_max_id = data['ID'].max() if not data.empty else 0

# 保存数据到文件
def save_data():
    feather.write_dataframe(data, data_file)

# 增加数据
def add_row(direction, quantity,下单价,摩擦成本,止损百分比,市值,开仓市值,最大市值,最小市值):
    global data, current_max_id
    new_id = current_max_id + 1
    current_max_id = new_id  # 更新当前最大ID
    new_row = pd.DataFrame({'ID': new_id, '多单': 0, '空单': 0,'下单价':0}, index=[0])

    if direction == '多':
        new_row['多单'] = quantity
        new_row['下单价'] = 下单价
        new_row['摩擦成本'] = 摩擦成本
        new_row['止损价']= 下单价*(1-止损百分比)
        new_row['市值']=市值
        new_row['开仓市值']=开仓市值
        new_row['最大市值']=最大市值
        new_row['最小市值']=开仓市值
    elif direction == '空':
        new_row['空单'] = quantity
        new_row['下单价'] = 下单价
        new_row['摩擦成本'] = 摩擦成本
        new_row['止损价']=下单价*(1+止损百分比)
        new_row['市值']=市值
        new_row['开仓市值']=开仓市值
        new_row['最大市值']=最大市值
        new_row['最小市值']=开仓市值
    else:
        print("方向错误，请输入'多'或'空'")
        return

    # 添加新行到DataFrame
    data = pd.concat([data,new_row], ignore_index=True)
    save_data()

# 删除指定的行
def delete_row(row_id):
    global data
    data = data[data['ID'] != row_id].reset_index(drop=True)
    save_data()

# 更新行
def update_row(row_id, direction, quantity):
    global data
    if row_id not in data['ID'].values:
        print("指定的ID不存在")
        return

    if direction == '多':

        data.loc[data['ID'] == row_id, '多单'] = quantity


    elif direction == '空':

        data.loc[data['ID'] == row_id, '空单'] = quantity


    else:
        print("方向错误，请输入'多'或'空'")
        return

    save_data()


def update_市值(row_id, 新市值):
    global data
    if row_id not in data['ID'].values:
        print("指定的ID不存在")
        return

    data.loc[data['ID'] == row_id, '市值'] = 新市值

    save_data()

def update_最大市值(row_id, 新市值):
    global data
    if row_id not in data['ID'].values:
        print("指定的ID不存在")
        return

    data.loc[data['ID'] == row_id, '最大市值'] = 新市值

    save_data()

def update_最小市值(row_id, 新市值):
    global data
    if row_id not in data['ID'].values:
        print("指定的ID不存在")
        return

    data.loc[data['ID'] == row_id, '最小市值'] = 新市值

    save_data()


def update_开仓市值(row_id, 新市值):
    global data
    if row_id not in data['ID'].values:
        print("指定的ID不存在")
        return

    data.loc[data['ID'] == row_id, '开仓市值'] = 新市值

    save_data()


# 查询数据
def query_data():
    return data

# 汇总数据
def summarize_data():
    return data[['多单', '空单']].sum()

# 重置数据
def reset_data():
    global data, current_max_id
    data = pd.DataFrame(columns=columns)
    current_max_id = 0  # 重置当前最大ID
    save_data()

# 示例操作
if __name__ == "__main__":
    # 查看初始数据
    print("初始数据：\n", query_data())




    # # 添加行
    # add_row('多', 10)
    # add_row('空', 5)
    #
    # print("添加行后的数据：\n", query_data())
    #
    # # 修改数据
    # update_row(1, '多', 15)  # 假设ID为1的行存在
    #
    # print("更新行后的数据：\n", query_data())
    #
    # # 删除数据
    # delete_row(2)  # 假设ID为2的行存在
    #
    # print("删除行后的数据：\n", query_data())

    # 重置数据
    # reset_data()

    print("重置数据后的数据：\n", query_data())
