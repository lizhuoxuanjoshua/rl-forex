import math

def calculate_stop_loss_threshold(max_market_value, opening_market_value, a=1, b=1, c=0.7):
    # Calculate the multiplier as the ratio of max market value to opening market value
    multiplier = (max_market_value / opening_market_value)

    # For every increase in multiplier by 1, the stop loss threshold increases by 1.0,
    # but remains constant within that multiplier interval
    return abs((multiplier - a) * b + c)


# Example usage
if __name__ == "__main__":
    for i in range(10):
        max_market_value = (i+1) * 10
        opening_market_value = 10  # Example value, replace with actual value
        stop_loss_threshold = calculate_stop_loss_threshold(max_market_value, opening_market_value)
        print(f"Opening Market Value: ${opening_market_value}, Profit Market Value: ${max_market_value}, Stop Loss Threshold: ${stop_loss_threshold*opening_market_value}")