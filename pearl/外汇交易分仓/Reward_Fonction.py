import numpy as np
from numba import jit
from tqdm import tqdm

@jit(nopython=True)
def find_max_parts(arr):
    if len(arr) <= 1:
        return ""

    mid = len(arr) // 2
    left = arr[:mid + 1] if len(arr) % 2 != 0 else arr[:mid]
    right = arr[mid:]

    left_sorted = sorted(left, reverse=True)
    right_sorted = sorted(right, reverse=True)

    for l, r in zip(left_sorted, right_sorted):
        if l > r:
            return "0" + find_max_parts(right)
        elif r > l:
            return "1" + find_max_parts(right)

    return "0" + find_max_parts(right)
def binary_position(array):
    # The logic here is not simply [first half, second half] (with the peak in the first half)
    # or [first half of the second half, second half of the second half] (with the peak in the latter half) -> outputting 01.
    # Instead, after outputting -> 01, it reverses, meaning it first calculates [first half of the second half, second half of the second half],
    # and then [first half, second half].
    # If the highest values on both sides are equal, it compares the second highest values, and so on.
    # If all values are equal on both sides, the left side is chosen.
    if len(array) <= 1:
        return ["", ""]

    array= np.array(array).astype(np.float64)

    result = find_max_parts(array)
    max_binary = result[::-1]
    max_binary = [int(char) for char in max_binary]

    result = find_max_parts(np.multiply(-1, array))
    min_binary  = result[::-1]
    min_binary  = [int(char) for char in min_binary ]
    return [max_binary, min_binary ]



def calculate_current_value_binary_position(data):

    current_visible = data
    # print(current_visible)
    # print("----------------current_visible_simi------------------")
    current_visible_simi = data[(len(data) + 1) // 2:]
    # print(current_visible_simi)

    high_binary, low_binary = binary_position(current_visible)
    high_binary_simi, low_binary_simi = binary_position(current_visible_simi)

    # print("----------------high_binary------------------")
    # print(high_binary)
    # print("----------------二分低------------------")
    # print(二分低)
    # print("----------------high_binary_simi------------------")
    # print(high_binary_simi)
    # print("----------------low_binary_simi------------------")
    # print(low_binary_simi)


    return high_binary, low_binary, high_binary_simi, low_binary_simi

def Reward_Increment_Based_on_Binary_Position(binary_list, 底数):

    """
    Convert a binary list to decimal value

    Parameters:
    binary_list (list): A list of binary digits (0s and 1s).

    Returns:
    int: The decimal representation of the binary list.
    """

    if 底数==1:
        decimal_value = 0
        for index, bit in enumerate(binary_list):
            decimal_value += bit * (index * 2)
        return decimal_value

    if 底数>-2:
        decimal_value = 0
        for index, bit in enumerate(binary_list):
            decimal_value += bit * (底数 ** index)
        return decimal_value


#test
if __name__ == '__main__':
    high_binary, low_binary ,high_binary_simi, low_binary_simi = calculate_current_value_binary_position([1, 4, 2, 81, 21, 45, 35, 45, 21, 45, 45, 85, 42, 48, 72, 48, 2, 4, 89, 78, 51, 54, 2, 54, 45, 62, 5, 3, 7, 6, 13, 2, 28, 30, 16])
    print(f"high_binary:{high_binary}")
    print(f"low_binary:{low_binary}")
    print(f"high_binary_simi:{high_binary_simi}")
    print(f"low_binary_simi:{low_binary_simi}")

