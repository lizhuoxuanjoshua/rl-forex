def calculate_friction_cost(traded_lots, leverage_ratio, friction_cost_percentage, current_price):
    contract_size = 100000
    margin_required = (traded_lots * contract_size * current_price) / leverage_ratio
    friction_cost = margin_required * (friction_cost_percentage / 100)

    return friction_cost


if __name__ == "__main__":
    # Example usage
    traded_lots = 0.16 # Trading 0.05 lots
    leverage_ratio = 20  # 20:1 leverage
    friction_cost_percentage = 1.5  # 1.5%
    current_price = 1.08381 # Current price of EUR/USD

    # Calculate used margin
    used_margin = calculate_friction_cost(traded_lots, leverage_ratio, friction_cost_percentage, current_price)
    print(f"Used margin: {used_margin:.2f} EUR")
