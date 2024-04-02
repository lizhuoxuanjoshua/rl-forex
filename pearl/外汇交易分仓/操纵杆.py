from pearl.外汇交易分仓 import OrderTable

# Function to adjust the net position to a target value
def adjust_net_position(订单表, 目标多空差):
    # Function to calculate the current net position difference between longs and shorts
    def 计算当前多空差(订单表):
        当前多单总体积 = 订单表['多单'].sum()  # Sum of all long positions
        当前空单总体积 = 订单表['空单'].sum()  # Sum of all short positions
        return 当前多单总体积 - 当前空单总体积  # Net position (long - short)

    当前多空差值 = 计算当前多空差(订单表)  # Current net position difference
    需要调整的差值 = 目标多空差 - 当前多空差值  # Adjustment needed to reach the target net position
    操作列表 = []  # List to store operations

    # Loop to adjust the net position until it's close enough to the target
    while abs(需要调整的差值) > 1e-8:
        # Prioritize closing larger long positions if the adjustment needed is negative
        if 需要调整的差值 < 0:
            订单表 = 订单表.sort_values(by='多单', ascending=False)  # Sort by long position size in descending order
            for index, row in 订单表.iterrows():
                if row['多单'] > 0 and 需要调整的差值 < 0:
                    平仓量 = min(row['多单'], abs(需要调整的差值))  # Closing volume
                    操作列表.append(['平多单', round(平仓量, 2), row['ID']])  # Add operation to close long
                    订单表.loc[订单表['ID'] == row['ID'], '多单'] -= 平仓量  # Update the table
                    需要调整的差值 += 平仓量  # Update the adjustment needed

        # Prioritize closing larger short positions if the adjustment needed is positive
        elif 需要调整的差值 > 0:
            订单表 = 订单表.sort_values(by='空单', ascending=False)  # Sort by short position size in descending order
            for index, row in 订单表.iterrows():
                if row['空单'] > 0 and 需要调整的差值 > 0:
                    平仓量 = min(row['空单'], 需要调整的差值)  # Closing volume
                    操作列表.append(['平空单', round(平仓量, 2), row['ID']])  # Add operation to close short
                    订单表.loc[订单表['ID'] == row['ID'], '空单'] -= 平仓量  # Update the table
                    需要调整的差值 -= 平仓量  # Update the adjustment needed

        # If there's still a difference after closing positions, open new ones
        if abs(需要调整的差值) > 1e-8:
            if 需要调整的差值 > 0:
                操作列表.append(['开多单', round(需要调整的差值, 2)])  # Open new long positions
                需要调整的差值 = 0
            elif 需要调整的差值 < 0:
                操作列表.append(['开空单', round(abs(需要调整的差值), 2)])  # Open new short positions
                需要调整的差值 = 0

    return 操作列表


# Test code
if __name__ == '__main__':
    当前订单流 = OrderTable.query_data()  # Current order flow
    print(当前订单流)

    当前多单汇总 = OrderTable.summarize_data()[0]  # Summary of long positions
    当前空单汇总 = OrderTable.summarize_data()[1]  # Summary of short positions

    # print(当前多单汇总)
    # print(当前空单汇总)
    print(f"当前多空差{当前多单汇总 - 当前空单汇总}")  # Print current net
