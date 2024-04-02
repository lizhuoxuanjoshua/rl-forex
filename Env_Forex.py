import json
import random
from logging.handlers import RotatingFileHandler
from time import sleep

import pandas as pd
import gym
import torch
from gym import spaces
import numpy as np
import feather
from matplotlib.animation import FuncAnimation
from scipy import stats
from torch import softmax

from pearl.api.action_result import ActionResult
from pearl.utils.instantiations.spaces.box_action import BoxActionSpace

import matplotlib.pyplot as plt
import mplfinance as mpf

from pearl.外汇交易分仓 import OrderTable, calculate_tradable_lots
from pearl.外汇交易分仓.Calculating_the_stop_loss_criterion import calculate_stop_loss_threshold
from pearl.外汇交易分仓.OrderTable import update_市值, update_最大市值
from pearl.外汇交易分仓.Reward_Fonction import calculate_current_value_binary_position, \
    Reward_Increment_Based_on_Binary_Position
from pearl.外汇交易分仓.Sorted_Comparison_Binary_Positioning_Method import calculate_current_value_binary_position1
from pearl.外汇交易分仓.calculate_friction_cost import calculate_friction_cost
from pearl.外汇交易分仓.calculate_used_margin import calculate_used_margin
from pearl.外汇交易分仓.operater import operater
from pearl.外汇交易分仓.操纵杆 import adjust_net_position

import logging

from pearl.外汇交易分仓.研究虚拟新高 import 判断并生成虚拟新高点


set_Account_net_value = 900
reward_countdown_setting=5
global action_countdown
action_countdown=20

def normalize_min_max(data):
    """
      Scale data to the [0, 1] range using min-max normalization.

      Parameters:
      data (list or numpy array): The data to be normalized.

      Returns:
      normalized_data (numpy array): The normalized data.
      """
    try:
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            # Handling the case where max and min values are equal
            normalized_data = np.full(len(data),0.5)  # 或者可以设置为1或其他适当的值
        else:
            normalized_data = (data - min_val) / (max_val - min_val)

        return normalized_data
    except:
        return []

class ForexTradingEnv(gym.Env):
    def __init__(self, csv_file_path, indicator_column_names):

        # Read CSV file==========================================================================
        self.data = pd.read_csv(csv_file_path)
        # Adding a Logger========================================================================
        self.logger = logging.getLogger('ForexTradingEnv')
        self.logger.setLevel(logging.INFO)  # Set the logging level to logging INFO will show the log output
        # Ensure that log messages do not propagate to the root logger, so they do not automatically print to the console
        self.logger.propagate = True

        # Create a log file processor
        log_file_path = 'pearl/外汇交易分仓/ForexTradingEnv.log'
        file_handler = RotatingFileHandler(log_file_path, maxBytes=1024 * 1024 * 5, backupCount=5)
        file_handler.setLevel(logging.INFO)


        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
        # ====================================================================================

        # ===========Insert basic market data and binary indicators========================================================

        self.indicator_column_names = indicator_column_names
        self.close = self.data["Close"].values
        self.high = self.data["High"].values
        self.low = self.data["Low"].values
        self.open = self.data["Open"].values


        # Attempting to convert the 'time' column to date time format, converting non compliant formats to NaT
        self.data['time'] = pd.to_datetime(self.data['time'], format='%Y%m%d %H%M%S', errors='coerce')
        # Delete rows containing NaT
        self.data = self.data.dropna(subset=['time'])
        # Extract date and time
        self.date = self.data['time'].dt.date
        self.time = self.data['time'].dt.time



        self.leverage_ratio = 200
        self.Stop_loss_percentage=0.02

        '''
        The friction cost mainly includes commission and spread, 
        and spread is very important! When the spread is below 45, 
        the basic friction cost can be covered. 
        If the spread is one or two hundred, the friction cost cannot be covered by 10 times
        
        '''
        self.friction_cost = 1  #
        self.Number_of_newly_added_state_spaces = 162
        self.Contract_size_per_lot = 100000
        operater("重置")
        self.stateShapeLength = len(indicator_column_names) + self.Number_of_newly_added_state_spaces

        # ===========Transaction records========================================================
        self.free_margin = set_Account_net_value

        self.Accumulated_rewards=0
        self.Accumulated_delayed_rewards=0
        self.Countdown_to_reward_redemption=reward_countdown_setting

        self.Previous_direction= "多"
        self.Same_direction_counting=0
        self.Bullet_size=0
        self.Relative_account_size=0
        self.股价净值差和=[]
        self.Number_of_sub_positions=10
        self.List_of_accumulated_rewards=[]
        self.List_of_net_asset_value_changes = []

        self.List_of_price=[]
        self.List_of_sub_position=[]
        self.Continuous_short_position_time=0

        self.Action_execution_countdown=action_countdown
        self.Action_execution_countdown_list=[]
        self.Previous_execution_action=0

        # State space and action space initialization===========================================================
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(indicator_column_names) + self.Number_of_newly_added_state_spaces,),
            dtype=np.float32
        )
        self.action_space = BoxActionSpace(0, 1)

        # Initialize other required variables
        self.current_step = 0
        self.state = None

    def step(self, action):

        # Update current steps=======================================================================
        self.current_step += 4

        # Clarify the past and future#####################################################################
        Decision_steps_have_been_made = self.current_step - 1
        A_step_in_the_future = self.current_step

        Previous_price = self.close[Decision_steps_have_been_made]
        Price_long = self.high[A_step_in_the_future]
        Price_short = self.low[A_step_in_the_future]

        Price_to_close_long = self.low[A_step_in_the_future]
        Price_to_close_short = self.high[A_step_in_the_future]

        This_date=self.date[Decision_steps_have_been_made]
        Thise_time=self.time[Decision_steps_have_been_made]


        # test==============================================================================
        # print(self.data.iloc[Decision_steps_have_been_made])
        # print(action)
        # print(self.current_step)
        # sleep(2)

        # ============================================================
        '''
        The action of the current step has already been executed before calling the 'step' method and calculating the 'nextunobservation'.
        This means that when we obtain the ` nextunobservation `, it reflects the environmental state after executing the previous action.
        This new observation state is not intended to change actions that have already occurred, but to analyze the current situation and guide the selection of next actions.
        Therefore, each execution of the 'step' is based on the current state and action, and then the environment is updated to the next state, providing a basis for decision-making on the next action.
        '''



        #Generate action===================================

        Action_preprocesed = action.cpu().numpy()[0]
        Action=self.map_value_to_range(Action_preprocesed,0,1,-1, 1)

        # print(Action)

        if self.Action_execution_countdown==0:
            self.Action_execution_countdown = action_countdown
            list_of_actions = self.Action_execution_countdown_list
            weights = np.linspace(1, 10, num=len(list_of_actions))  # Adjust weights to match slice length
            # Now use the slice and weights of matching size
            Action = np.average(list_of_actions, weights=weights)
            Action = ((Action * 100) * 5) / 100
            self.Previous_execution_action=Action
            list_of_actions_copy=self.Action_execution_countdown_list
            self.Action_execution_countdown_list=[]
        else:
            #Give intelligent agents the opportunity to break through the limitations of the countdown
            self.Action_execution_countdown-=1
            self.Action_execution_countdown_list.append(round(Action, 1))
            Action=0


        OrderTable_copy = OrderTable.query_data()
        Order_market_value=OrderTable_copy["市值"].sum()
        Total_market_value= self.free_margin + Order_market_value


        # Update calculation of floating profit and loss and net value changes in order flow===========================================================================================

        OrderTable_copy = OrderTable.query_data()
        for i in range(len(OrderTable_copy)):
            # Determine whether the order is long or short

            Long = float(OrderTable_copy.at[i, "多单"] > 0)
            Short = float(OrderTable_copy.at[i, "空单"] > 0)
            Open_price = float(OrderTable_copy.at[i, "下单价"])

            Open_value = float(OrderTable_copy.at[i, "开仓市值"])
            value = float(OrderTable_copy.at[i, "市值"])
            max_value = float(OrderTable_copy.at[i, "最大市值"])

            Open_price = Open_price

            # Judgment of stop loss criteria=====================================================================================================
            Stop_loss_criteria = calculate_stop_loss_threshold(max_value, Open_value)

            # Update the market value of the current order
            if Long:
                平仓ID = OrderTable_copy.at[i, "ID"]
                volume = round(float(OrderTable_copy.at[i, "多单"]), 2)

                profit = (self.close[A_step_in_the_future] - Open_price) * float(volume) * self.Contract_size_per_lot

                New_value = Open_value + profit
                update_市值(平仓ID, New_value)

                max_value = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '最大市值'].values[0])
                if New_value > max_value:
                    update_最大市值(平仓ID, New_value)

                # Determine the position to be pushed out============================================================
                if New_value < Open_value * Stop_loss_criteria:
                    print(
                        f'Maximum market value：{round(max_value)}，Current market value：{round(New_value)}Less than the opening market value：Forced to be pushed out')

                    direction = "平"
                    Original_volume = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '多单'].values[0])
                    value = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(direction, volume, 平仓ID)
                    value_when_closed = value * (volume / Original_volume)
                    self.free_margin += value_when_closed
                    self.List_of_sub_position.append(self.Bullet_size)
                    print(f"Bullets recalled, scale{self.Bullet_size}")
                # if 新市值>开仓市值*2:
                #     print('订单市值大于开仓市值的200%，被止盈')
                #
                #     方向 = "平"
                #     原始数量 = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '多单'].values[0])
                #     市值 = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '市值'].values[0])
                #     operater(方向, 数量, 平仓ID)
                #     平仓市值 = 市值 * (数量 / 原始数量)
                #     self.free_margin += 平仓市值

            if Short:
                平仓ID = OrderTable_copy.at[i, "ID"]
                volume = round(float(OrderTable_copy.at[i, "空单"]), 2)
                profit = -(self.close[A_step_in_the_future] - Open_price) * float(volume) * self.Contract_size_per_lot

                New_value = Open_value + profit
                update_市值(平仓ID, New_value)

                max_value = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '最大市值'].values[0])
                if New_value > max_value:
                    update_最大市值(平仓ID, New_value)

                # Determine the position to be pushed out============================================================
                # The situation of failure on the first attempt=====================================================================
                if New_value < Open_value * Stop_loss_criteria:
                    print(
                        f'Maximum market value：{round(max_value)}，Current market value：{round(New_value)}Less than the opening market value：{round(Open_value)}Forced to be pushed out')

                    direction = "平"
                    Original_volume = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '空单'].values[0])
                    value = float(OrderTable_copy.loc[OrderTable_copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(direction, volume, 平仓ID)
                    value_when_closed = value * (volume / Original_volume)
                    self.free_margin += value_when_closed
                    print(f"Bullets recalled, scale{self.Bullet_size}，Recall front clip{self.List_of_sub_position}")
                    self.List_of_sub_position.append(self.Bullet_size)
                    print(f"Recalled clip{self.List_of_sub_position}")


        #Order operation================================================================================================================================

        # Check if the size of the bullets is reasonable================================================================================================
        if len(self.List_of_sub_position) > 0:
            Number_of_magazine_clips = self.Number_of_sub_positions
            Magazine_size = self.List_of_sub_position[-1]

            friction_cost = calculate_friction_cost(Magazine_size, self.leverage_ratio, self.friction_cost, Price_long)
            value = calculate_used_margin(float(Magazine_size), self.leverage_ratio, self.friction_cost, Price_long)

            Cost_required_for_opening = (friction_cost + value) * Number_of_magazine_clips
            Cost_additionnal_for_opening = friction_cost * Number_of_magazine_clips

            Order_market_value = OrderTable_copy["市值"].sum()
            Total_market_value = self.free_margin + Order_market_value

            if Total_market_value < Cost_required_for_opening:
                print(f"The size of the magazine is unreasonable! Readjust, the cost required for opening{Cost_required_for_opening} ,Account net value {Total_market_value}")
                tradable_lots = calculate_tradable_lots.calculate_tradable_lots(self.free_margin + Order_market_value, self.leverage_ratio,
                                                                               self.friction_cost, Previous_price)
                volume_of_sub_positions = np.floor((tradable_lots / self.Number_of_sub_positions) * 100) / 100
                self.Bullet_size = volume_of_sub_positions
                分仓列表copy = self.List_of_sub_position
                self.List_of_sub_position = np.full(len(self.List_of_sub_position), self.Bullet_size).tolist()
                print(f"Readjust the magazine size! Before adjustment{分仓列表copy},After adjustment{self.List_of_sub_position}")


        # Determine if the action is triggered==================================================================================================

        # I know the action that has already occurred, but I don't know the corresponding joystick pushing distance before this action occurred, so I need to use the price from the previous step to calculate it
        # The operation has occurred ->the consequences have not been observed yet, and the net value at this time is a review of the previous action--------
        # Query the maximum number of available buyers===============================================================================================
        OrderTable_copy = OrderTable.query_data()
        Order_market_value=OrderTable_copy["市值"].sum()
        Total_market_value= self.free_margin
        # Total_market_value= self.free_margin + Order_market_value

        tradable_lots = calculate_tradable_lots.calculate_tradable_lots(Total_market_value, self.leverage_ratio, self.friction_cost, Previous_price)

        if abs(Action) > 0.3:

            type_of_action="开多" if Action>0 else "开空"

            print("###################################################################################################################")
            print(f"Action triggered! Action type={type_of_action}========================================================================================")

            # 输出信息======================================================================================================
            print(f"Action execution countdown{self.Action_execution_countdown}")
            print(f"Action execution countdown list{list_of_actions_copy}")
            print(f"Action{Action}")
            # print(Action)
            print(f"--current step:{Decision_steps_have_been_made}")

            # ====================================If the position list is  out==============================================
            if self.List_of_sub_position == []:
                '''
                Execute warehouse allocation plan and allocate the maximum available number of hands into 10 portions
                '''

                volume_of_sub_positions = round(tradable_lots / self.Number_of_sub_positions, 2)
                print(f"Current net value{self.free_margin + Order_market_value} 10 positions have been allocated, each with disposable{volume_of_sub_positions}")
                if volume_of_sub_positions < 0.01:
                    print("The current funds have not been recovered yet Waiting for operation")
                else:
                    # The current funds have not been fully depleted=========================================================================
                    # Push ten disposable funds into the sub warehouse list-----------------------------------------------------------------
                    self.List_of_sub_position = np.full(self.Number_of_sub_positions, volume_of_sub_positions).tolist()
                    self.Bullet_size = volume_of_sub_positions
                    self.Relative_account_size = self.free_margin + Order_market_value
                    print(f"Planned positions list{self.List_of_sub_position}lots")


            # Determine whether the account size has relatively doubled===================================================================================
            Total_market_value = self.free_margin + Order_market_value

            if Total_market_value > self.Relative_account_size * 2:
                volume_of_sub_positions = round(tradable_lots / self.Number_of_sub_positions, 2)
                self.Bullet_size = volume_of_sub_positions
                print(f"Congratulations on doubling your account size，from{self.Relative_account_size}to{Total_market_value}")
                self.Relative_account_size = Total_market_value

            if self.List_of_sub_position != [] and len(OrderTable_copy) <= 8:
                print(f"There are still bullets left to be fired{self.List_of_sub_position}lots")
                current_lots_of_sub_position = self.List_of_sub_position[-1]

                # The situation of opening long orders================================================================================================================



                if Action > 0 :

                    direction = "多"

                    volume = current_lots_of_sub_position
                    friction_cost = calculate_friction_cost(volume, self.leverage_ratio, self.friction_cost, Price_long)

                    value = calculate_used_margin(float(volume), self.leverage_ratio, self.friction_cost, Price_long)

                    if self.free_margin > value:
                        # ========================margin is enough======================================
                        self.free_margin -= friction_cost
                        self.free_margin -= value

                        operater(direction, volume, 下单价=Price_long, 摩擦成本=friction_cost,
                                 止损百分比=self.Stop_loss_percentage / self.leverage_ratio, 市值=value, 开仓市值=value)
                        print(f"Long opened{current_lots_of_sub_position}lots,friction_cost{friction_cost},value{value}")
                        self.Previous_direction = direction
                        self.List_of_sub_position.pop()
                    else:
                        print(
                            f'Insufficient available pre deposit! Order quantity{volume},need{value + friction_cost},rest{self.free_margin},value now{Total_market_value}')





                # The situation of opening short orders================================================================================================================
                if Action < 0:
                    direction = "空"
                    volume = current_lots_of_sub_position
                    friction_cost = calculate_friction_cost(volume, self.leverage_ratio, self.friction_cost, Price_short)
                    value = calculate_used_margin(float(volume), self.leverage_ratio, self.friction_cost, Price_short)

                    if self.free_margin > value:

                        self.free_margin -= friction_cost
                        self.free_margin -= value

                        operater(direction, volume, 下单价=Price_long, 摩擦成本=friction_cost,
                                 止损百分比=self.Stop_loss_percentage / self.leverage_ratio, 市值=value, 开仓市值=value)
                        print(f"short opened!{current_lots_of_sub_position}lots,friction_cost{friction_cost},value{value},now stop loss")
                        self.Previous_direction = direction
                        self.List_of_sub_position.pop()
                    else:
                        print(
                            f'Insufficient available pre deposit! Order quantity{volume},need{value + friction_cost},rest{self.free_margin},value now{Total_market_value}')



            print(f"-------Post operation position list - net value before change{Total_market_value}----")
            print(OrderTable.query_data())
        else:
            # 动作未被触发
            1


        #Total value obtained by updating net value+market value==================================================================================
        OrderTable_copy = OrderTable.query_data()
        Order_market_value=OrderTable_copy["市值"].sum()
        Total_market_value= self.free_margin + Order_market_value




        # print(f"账户净值{round(self.账户净值, 2)}")
        self.List_of_net_asset_value_changes.append(Total_market_value)
        self.List_of_price.append(self.close[A_step_in_the_future])
        # calculate the reward------------------------------------------------

        '''
        The order flow operation has been completed in this step, and then the pending step (until the price is unknown) summarizes this step.
        So as to know if this step is done correctly, it is required that after the current calculation is completed,
        That is to say, the calculation starts at the exact end of one minute, and the order is placed in one second. All actions are completed in the last second of this step,
        '''

        reward = self._calculate_reward(action, 0)

        # observe============================================================================
        self.state = self._next_observation()

        # terminated===============================================================
        terminated = False
        truncated = False  # 根据需要设置
        if self.current_step >= len(self.data)-1 or Total_market_value <= 120:
            terminated = True
            truncated = False
            action_result = ActionResult(
                observation=np.zeros_like(self.state).astype(np.float32),
                reward=np.float32(0.0),
                terminated=terminated,
                truncated=truncated,
                info={}
            )

        # pas terminated===============================================================
        else:
            action_result = ActionResult(
                observation=self.state.astype(np.float32),
                reward=np.float32(reward),
                terminated=terminated,
                truncated=truncated,
                info={}
            )

            # print(action_result)
        return action_result

    def reset(self, seed=None):

        self.current_step = 0
        self.state = self._next_observation()
        # 重置账户净值========================================================
        # ===========交易记录========================================================
        self.free_margin = set_Account_net_value

        self.Accumulated_rewards=0
        self.Accumulated_delayed_rewards=0
        self.Countdown_to_reward_redemption=reward_countdown_setting

        self.List_of_sub_position=[]
        self.List_of_net_asset_value_changes = []

        self.List_of_accumulated_rewards=[]

        self.List_of_price=[]

        self.Continuous_short_position_time=0

        self.Number_of_sub_positions=10
        self.股价净值差和=[]

        self.Previous_direction= "多"
        self.Same_direction_counting=0

        self.Bullet_size=0
        self.Relative_account_size=0

        self.Action_execution_countdown=action_countdown
        self.Action_execution_countdown_list=[]
        self.Previous_execution_action=0
        # 状态空间和动作空间初始化===========================================================

        operater("重置")
        return self.state, {}

    def map_value_to_range(self, value, ori_min,ori_max,new_min, new_max):

        # 原始范围的最小值和最大值
        original_min = ori_min
        original_max = ori_max

        # 映射公式
        mapped_value = ((value - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min

        return mapped_value

    def _calculate_reward(self, action, 上一个净值):


        # 二分奖励计算部分=======================================================================================
        二分奖励计算列表 = 判断并生成虚拟新高点(np.array(self.List_of_net_asset_value_changes).astype(np.float64))
        二分高, 二分低, 二分高simi, 二分低simi = calculate_current_value_binary_position(二分奖励计算列表)

        二分高奖励 = Reward_Increment_Based_on_Binary_Position(二分高,2)
        二分高simi奖励 = Reward_Increment_Based_on_Binary_Position(二分高simi,2) / 2
        二分低惩罚 = Reward_Increment_Based_on_Binary_Position(二分低,2)
        二分低simi惩罚 = Reward_Increment_Based_on_Binary_Position(二分低simi,2) / 2

        reward_sum = 二分高奖励 + 二分高simi奖励
        punnish = 二分低惩罚 + 二分低simi惩罚

        # For now, only rewards will be given without punishment,
        # because intelligent agents will actively choose to lose all their capital
        # as soon as possible to avoid punishment

        Current_survival_steps=len(self.List_of_net_asset_value_changes)

        reward = (reward_sum)/10000*(Current_survival_steps*0.01)

        # print(self.Continuous_short_position_time)
        self.Accumulated_rewards= self.Accumulated_rewards + reward
        self.List_of_accumulated_rewards.append(self.Accumulated_rewards)

        return reward


    def _next_observation(self):

        # ================Read binary indicators=====================================================
        当前dataframe=  self.data.iloc[0:self.current_step]
        obs = self.data[self.indicator_column_names].iloc[self.current_step]  # self.current_step是未来的一步，所以这里是未来的一步的二分指标
        obs = obs.values.astype(np.float32)
        normoized_value = normalize_min_max(self.List_of_net_asset_value_changes)
        normoized_price = normalize_min_max(self.List_of_price)



        difference = np.subtract(normoized_value, normoized_price)
        difference_accumulated = np.sum(difference)
        self.股价净值差和.append(difference_accumulated)
        try:
            二分高, 二分低, 二分高simi, 二分低simi = calculate_current_value_binary_position1(np.array(self.List_of_net_asset_value_changes))
        except:
            二分高 = np.full(20, 0)
            二分低 = np.full(20, 0)
            二分高simi = np.full(20, 0)
            二分低simi = np.full(20, 0)

        try:
            二分高x, 二分低x, 二分高simix, 二分低simix = calculate_current_value_binary_position1(np.array( self.股价净值差和))
        except:
            二分高x = np.full(20, 0)
            二分低x = np.full(20, 0)
            二分高simix = np.full(20, 0)
            二分低simix = np.full(20, 0)

        #Add these four lists to the observation space and turn them into one list
        obs = np.append(obs, 二分高)
        obs = np.append(obs, 二分低)
        obs = np.append(obs, 二分高simi)
        obs = np.append(obs, 二分低simi)


        obs = np.append(obs, 二分高x)
        obs = np.append(obs, 二分低x)
        obs = np.append(obs, 二分高simix)
        obs = np.append(obs, 二分低simix)


        obs=np.append(obs, self.Action_execution_countdown / 100)
        obs=np.append(obs, self.Same_direction_counting / 10)

        return obs

    def render(self, mode='human'):
        if mode == 'human':
            if mode == 'human':
                with open('pearl/外汇交易分仓/动态监控.json', 'w') as file:
                    json.dump(self.List_of_net_asset_value_changes, file)
                with open('pearl/外汇交易分仓/奖励动态监控.json', 'w') as file:
                    json.dump(self.List_of_accumulated_rewards, file)


if __name__ == '__main__':
    pass

