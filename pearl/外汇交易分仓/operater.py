from pearl.外汇交易分仓 import OrderTable
from pearl.外汇交易分仓.OrderTable import update_市值, update_开仓市值


def operater(方向, 数量=None, 索引=None,下单价=None,摩擦成本=None,止损百分比=None,市值=None,开仓市值=None,最大市值=0,最小市值=0):
    操作结果 = "操作失败"
    之前下单价格=0
    if 方向 == '多' or 方向 == '空':
        OrderTable.add_row(方向, 数量,下单价,摩擦成本=摩擦成本,止损百分比=止损百分比,市值=市值,开仓市值=开仓市值,最大市值=最大市值,最小市值=最小市值)
        操作动作 = "买入" if 方向 == '多' else "卖出"
        操作结果 = f"{操作动作}{数量}手，下单价{下单价}，市值{市值}，开仓市值{开仓市值}"
    elif 方向 == '平':
        # 需要先找到ID对应的行
        row = OrderTable.query_data().loc[OrderTable.query_data()['ID'] == 索引]
        if row.empty:
            return "指定的索引不存在"

        该索引数量 = round(row[['多单', '空单']].sum(axis=1).values[0],2) #这里出现了持仓数量0.0099999999的情况，所以要暂时round一下
        该索引市值 = row['市值'].values[0]
        该索引开仓市值 = row['开仓市值'].values[0]



        if float(数量) > float(该索引数量):
            print(f"平仓数量{数量}大于持仓数量{该索引数量}")
            assert False, "平仓数量大于持仓数量"

        if round(float(该索引数量),2) == round(float(数量),2):
            OrderTable.delete_row(索引)
            操作结果 = f"平仓ID{索引}，{该索引数量}手"
        else:
            修改后的数量 = float(该索引数量) - float(数量)
            数量变化比例=修改后的数量/该索引数量
            修改后的市值=该索引市值*数量变化比例
            修改后的开仓市值=该索引开仓市值*数量变化比例
            # 需要判断是多单还是空单
            if row['多单'].values[0] > 0:
                之前下单价格=OrderTable.update_row(索引, '多', 修改后的数量)
                update_市值(索引,修改后的市值)
                update_开仓市值(索引,修改后的开仓市值)

            else:
                之前下单价格=OrderTable.update_row(索引, '空', 修改后的数量)
                update_市值(索引,修改后的市值)
                update_开仓市值(索引,修改后的开仓市值)
            操作结果 = f"平仓ID{索引}，{数量}手"
    elif 方向 == "重置":
        OrderTable.reset_data()
        操作结果 = "重置订单表"

    return 操作结果




#主程序入口
if __name__ == "__main__":

    print(f"操作前{OrderTable.query_data()}" )

    print(operater("平",0.23,770))

    print(f"操作后{OrderTable.query_data()}" )