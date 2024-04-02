def calculate_tradable_lots(available_capital, leverage_ratio, friction_cost_percentage, current_price):
    contract_size = 100000  # Standard lot size in units of the base currency
    total_tradable_capital = available_capital * leverage_ratio
    estimated_lots = total_tradable_capital / (current_price * contract_size)
    estimated_margin_per_lot = (current_price * contract_size) / leverage_ratio
    friction_cost_per_lot = estimated_margin_per_lot * (friction_cost_percentage / 100)
    total_cost_per_lot = estimated_margin_per_lot + friction_cost_per_lot
    tradable_lots = available_capital / total_cost_per_lot
    tradable_lots = (tradable_lots // 0.01) * 0.01
    return tradable_lots


# 参数
available_capital = 899
leverage_ratio = 20
friction_cost_percentage = 1.5
current_price = 1.08381


if __name__ == '__main__':
    # 计算可买手数
    tradable_lots = calculate_tradable_lots(available_capital, leverage_ratio, friction_cost_percentage, current_price)
    print(f"Tradable lots: {tradable_lots:.2f}")
