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
from pearl.外汇交易分仓.二分净值观测 import 计算当下净值二分位置
from pearl.外汇交易分仓.判断止损标准 import 计算止损标准
from pearl.外汇交易分仓.判断止盈标准 import calculate_take_profit_threshold
from pearl.外汇交易分仓.奖励二分当下 import 计算当下二分位置, 根据二分位置递增奖励
from pearl.外汇交易分仓.手数计算占用保证金 import calculate_used_margin
from pearl.外汇交易分仓.手数计算摩擦成本 import calculate_friction_cost
import matplotlib.pyplot as plt
import mplfinance as mpf

from pearl.外汇交易分仓 import 保证金计算可买手数, 订单表
from pearl.外汇交易分仓.操作者 import operater
from pearl.外汇交易分仓.操纵杆 import adjust_net_position
import warnings
warnings.filterwarnings('ignore')

import logging

from pearl.外汇交易分仓.研究虚拟新高 import 判断并生成虚拟新高点
from pearl.外汇交易分仓.订单表 import update_市值, update_最大市值, update_最小市值

设定账户净值 = 9000
设定奖励兑付倒计时=5
global action_countdown
action_countdown=20

def normalize_min_max(data):
    """
    使用最小-最大归一化将数据缩放到 [0, 1] 范围。

    参数：
    data (list 或 numpy 数组)：要归一化的数据。

    返回值：
    normalized_data (numpy 数组)：归一化后的数据。
    """
    try:
        data = np.array(data)
        min_val = np.min(data)
        max_val = np.max(data)

        if min_val == max_val:
            # 处理最大值和最小值相等的情况
            normalized_data = np.full(len(data),0.5)  # 或者可以设置为1或其他适当的值
        else:
            normalized_data = (data - min_val) / (max_val - min_val)

        return normalized_data
    except:
        return []

class ForexTradingEnv(gym.Env):
    def __init__(self, csv_file_path, indicator_column_names,a,b,c,d,e,f,回测开始百分比,回测结束百分比):

        # 画图初始化==========================================================================
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.e=e
        self.f=f


        # 读取CSV文件==========================================================================
        self.data = pd.read_csv(csv_file_path)
        回测开始百分比=回测开始百分比/100
        回测结束百分比=回测结束百分比/100
        回测开始索引=int(round(len(self.data)*回测开始百分比))
        回测结束索引=int(round(len(self.data)*回测结束百分比))-1

        self.data=self.data[回测开始索引:回测结束索引]

        # 添加日志记录器========================================================================
        self.logger = logging.getLogger('ForexTradingEnv')
        self.logger.setLevel(logging.INFO)  # 设置日志级别 设置成logging.INFO就可以看到日志输出了
        # 确保日志消息不会传播到根记录器，这样就不会自动打印到控制台
        self.logger.propagate = True

        # 创建一个日志文件处理器
        log_file_path = 'ForexTradingEnv.log'
        file_handler = RotatingFileHandler(log_file_path, maxBytes=1024 * 1024 * 5, backupCount=5)
        file_handler.setLevel(logging.INFO)

        # 创建一个日志格式器，并设置给文件处理器,设置成logging.INFO就可以看到日志输出了
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到记录器
        if not self.logger.handlers:  # 避免重复添加处理器
            self.logger.addHandler(file_handler)
        # ====================================================================================

        # ===========插入行情接触数据和二分指标========================================================

        self.indicator_column_names = indicator_column_names
        self.close = self.data["Close"].values
        self.high = self.data["High"].values
        self.low = self.data["Low"].values
        self.open = self.data["Open"].values



        # 尝试将'time'列转换为日期时间格式，不符合格式的转换为NaT
        self.data['time'] = pd.to_datetime(self.data['time'], format='%Y%m%d %H%M%S', errors='coerce')
        # 删除包含NaT的行
        self.data = self.data.dropna(subset=['time'])
        # 提取日期和时间
        self.date = self.data['time'].dt.date
        self.time = self.data['time'].dt.time


        self.盈利加成倍数=30
        self.杠杆倍数 = 30
        self.止损百分比=0.02
        # self.止损百分比=self.止损百分比/self.杠杆倍数
        self.摩擦成本 = 1  # 摩擦成本主要包含佣金和点差,点差十分重要！当点差在45一下，基本这个摩擦成本能能涵盖，如果点差一两百，摩擦成本10倍都覆盖不了
        self.新增状态空间数量 = 162
        self.每手合约大小 = 100000
        operater("重置")
        self.stateShapeLength = len(indicator_column_names) + self.新增状态空间数量

        # ===========交易记录========================================================
        self.可用预存款 = 设定账户净值
        self.不计摩擦成本净值 = 设定账户净值
        self.只计盈利的净值 = 设定账户净值
        self.奖励累计=0
        self.延迟奖励累计=0
        self.奖励兑付倒计时=设定奖励兑付倒计时

        self.上一个方向="多"
        self.相同方向计数=0
        self.子弹规模=0
        self.账户相对规模=0

        self.分仓数=10


        self.观察净盈亏 = 0
        self.观察摩擦成本 = 0

        self.净值变化列表 = []
        self.不计摩擦成本净值变化列表 = []
        self.只计盈利的净值变化列表 = []
        self.奖励累计列表 = []
        self.股价列表=[]
        self.分仓列表=[]
        self.股价净值差和=[]
        self.连续空仓时间=0

        self.动作执行倒计时=action_countdown
        self.动作执行倒计时列表=[]
        self.上一个执行动作=0

        # 状态空间和动作空间初始化===========================================================
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(indicator_column_names) + self.新增状态空间数量,),
            dtype=np.float32
        )
        self.action_space = BoxActionSpace(0, 1)

        # 初始化其他所需变量
        self.current_step = 0
        self.state = None

    def step(self, action):

        # 更新当前步骤=======================================================================
        self.current_step += 4

        # 理清楚过去和未来#####################################################################
        已做决定步 = self.current_step - 1
        未来的一步 = self.current_step

        上一步价格 = self.close[已做决定步]
        开多考虑点差价格 = self.high[未来的一步]
        开空考虑点差价格 = self.low[未来的一步]

        平多考虑点差价格 = self.low[未来的一步]
        平空考虑点差价格 = self.high[未来的一步]

        这一步date=self.date[已做决定步]
        这一步time=self.time[已做决定步]

        # if 这一步time.hour == 23 and 这一步time.minute == 50:
        #     #print("当前时间是23:50")
        # #print("========================================================================================")
        # #print(f"date：{这一步date} time：{这一步time}")


        # test==============================================================================
        # #print(self.data.iloc[已做决定步])
        # #print(action)
        # #print(self.current_step)
        # sleep(2)

        # 操作已产生->观测后果并复盘============================================================
        '''
          在调用`step`方法并计算`_next_observation`之前，当前步骤的动作（`action`）已经被执行。
          这意味着，当我们获取`_next_observation`时，它反映的是执行上一动作之后的环境状态。
          这个新的观测状态不是用来改变已经发生的动作，而是用于分析当前的局势并指导下一步的动作选择。
          因此，每次`step`的执行都是基于当前的状态和动作，然后环境更新到下一个状态，为下一次动作的决策提供依据。
        '''

        # 总结过去==============================================================================

        #产生动作===================================






        # 打印信息==============================================================================
        # #print(
        #     f"#动作:{动作} #操作动作{round(动作 * 100)}%已经发生")

        订单表copy = 订单表.query_data()
        订单市值=订单表copy["市值"].sum()
        总值= self.可用预存款 + 订单市值
        # #print(f"---------操作前持仓列表---未变动前净值{总值}--------")
        # #print(订单表.query_data())
        # #print(f"最大可买手数{最大可买手数}")


        # 更新计算订单流浮动盈亏和净值变化===========================================================================================

        订单表copy = 订单表.query_data()
        for i in range(len(订单表copy)):
            # 判断该笔订单是多单还是空单

            多单 = float(订单表copy.at[i, "多单"] > 0)
            空单 = float(订单表copy.at[i, "空单"] > 0)
            下单价 = float(订单表copy.at[i, "下单价"])

            开仓市值 = float(订单表copy.at[i, "开仓市值"])
            市值 = float(订单表copy.at[i, "市值"])
            最大市值=float(订单表copy.at[i, "最大市值"])
            最小市值=float(订单表copy.at[i, "最小市值"])


            开仓价 = 下单价



            #止损标准的判断=====================================================================================================
            止损标准=计算止损标准(最大市值,开仓市值,self.a,self.b,self.c)
            止盈标准=calculate_take_profit_threshold(最小市值, 开仓市值, self.d, self.e, self.f)


            # 更新当前订单的市值
            if 多单:
                平仓ID = 订单表copy.at[i, "ID"]
                数量 = round(float(订单表copy.at[i, "多单"]), 2)

                平仓盈亏 = (self.close[未来的一步] - 开仓价) * float(数量) * self.每手合约大小

                新市值 = 开仓市值 + 平仓盈亏
                update_市值(平仓ID, 新市值)

                最大市值=float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '最大市值'].values[0])
                最小市值=float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '最小市值'].values[0])
                if 新市值>最大市值:
                    update_最大市值(平仓ID, 新市值)
                if 新市值<最小市值:
                    update_最小市值(平仓ID, 新市值)



                #判断推出仓位============================================================
                if 新市值<开仓市值*止损标准:
                    #print(f'最大市值：{round(最大市值)}，当前市值：{round(新市值)}小于开仓市值：{round(开仓市值)}的{(止损标准)}倍"，即小于{round(开仓市值 * 止损标准)}被强制推出')

                    方向="平"
                    原始数量 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '多单'].values[0])
                    市值 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(方向, 数量, 平仓ID)
                    平仓市值 = 市值 * (数量 / 原始数量)
                    self.可用预存款 += 平仓市值
                    self.分仓列表.append(self.子弹规模)

                elif 新市值 > 开仓市值 * 止盈标准:
                    # print(f'最大市值：{round(最大市值)}，当前市值：{round(新市值)}小于开仓市值：{round(开仓市值)}的{(止损标准)}倍"，即小于{round(开仓市值 * 止损标准)}被强制推出')

                    方向 = "平"
                    原始数量 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '多单'].values[0])
                    市值 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(方向, 数量, 平仓ID)
                    平仓市值 = 市值 * (数量 / 原始数量)
                    self.可用预存款 += 平仓市值
                    self.分仓列表.append(self.子弹规模)


            if 空单:
                平仓ID = 订单表copy.at[i, "ID"]
                数量 = round(float(订单表copy.at[i, "空单"]), 2)
                平仓盈亏 = -(self.close[未来的一步] - 开仓价) * float(数量) * self.每手合约大小

                新市值 = 开仓市值 + 平仓盈亏
                update_市值(平仓ID, 新市值)


                最大市值=float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '最大市值'].values[0])
                最小市值=float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '最小市值'].values[0])

                if 新市值>最大市值:

                    # print(111111111111111111111)
                    # print(f"新市值>最大市值！更新最大市值，新市值{新市值},最大市值{最大市值}")
                    update_最大市值(平仓ID, 新市值)

                if 新市值<最小市值:
                    # print(2222222222222222222222222222)
                    # print(f"新市值<最小市值！更新最小市值，新市值{新市值},最小市值{最小市值}")
                    update_最小市值(平仓ID, 新市值)


                # 判断推出仓位============================================================
                #第一次就失败的情况=====================================================================

                if 新市值 < 开仓市值*止损标准:
                    # print(f'最大市值：{round(最大市值)}，当前市值：{round(新市值)}小于开仓市值：{round(开仓市值)}的{(止损标准)}倍"，即小于{round(开仓市值 * 止损标准)}被强制推出')

                    方向 = "平"
                    原始数量 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '空单'].values[0])
                    市值 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(方向, 数量, 平仓ID)
                    平仓市值 = 市值
                    self.可用预存款 += 平仓市值
                    #print(f"子弹被召回，规模{self.Bullet_size}，召回前弹夹{self.List_of_sub_position}")
                    self.分仓列表.append(self.子弹规模)

                elif 新市值 > 开仓市值 * 止盈标准:
                    # print(f'最大市值：{round(最大市值)}，当前市值：{round(新市值)}小于开仓市值：{round(开仓市值)}的{(止损标准)}倍"，即小于{round(开仓市值 * 止损标准)}被强制推出')

                    方向 = "平"
                    原始数量 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '多单'].values[0])
                    市值 = float(订单表copy.loc[订单表copy['ID'] == 平仓ID, '市值'].values[0])
                    operater(方向, 数量, 平仓ID)
                    平仓市值 = 市值
                    self.可用预存款 += 平仓市值
                    self.分仓列表.append(self.子弹规模)

    #订单操作模块================================================================================================================================






        # 判断动作被触发==================================================================================================


        # 知道了已经发生的动作，但是不知道这个发生这个动作前对应的操纵杆推动距离，所以要用上一步的价格来计算
        # 操作已产生->尚未观测后果，此时的净值是上一次的行动复盘--------
        #查询最大可买手数===============================================================================================
        订单表copy = 订单表.query_data()
        订单市值=订单表copy["市值"].sum()
        总值= self.可用预存款
        # 总值= self.free_margin + 订单市值

        最大可买手数 = 保证金计算可买手数.calculate_tradable_lots(总值, self.杠杆倍数, self.摩擦成本, 上一步价格)


        if 1:

            # ====================================如果仓位列表被打出去==============================================
            if self.分仓列表 == []:
                # 执行分仓规划，将最大的可用手数分配成10份=============================================================
                每份可支配手数 = round(最大可买手数 / self.分仓数, 2)
                #print(f"当前净值{self.free_margin + 订单市值} 已分配10份仓位，每份可支配手数{每份可支配手数}")
                if 每份可支配手数 < 0.01:
                    #print("当前资金尚未回笼.....等待操作")
                    1
                else:
                    # 当前资金没有被亏完=========================================================================
                    # 将十份可支配资金推入分仓列表-----------------------------------------------------------------
                    self.分仓列表 = np.full(self.分仓数, 每份可支配手数).tolist()
                    self.子弹规模=每份可支配手数
                    self.账户相对规模=self.可用预存款 + 订单市值
                    #print(f"已规划分仓列表{self.List_of_sub_position}手")
                    # input("我要开始推仓位了")

            #判断账户规模是否相对翻倍===================================================================================
            总值= self.可用预存款 + 订单市值


            if self.分仓列表 != []  and len(订单表copy)<=8:
                #print(f"还有没打完的子弹{self.List_of_sub_position}手")
                当前子弹手数 = self.分仓列表[-1]

                # 开多单的情况================================================================================================================

                #判断同时持有的订单数不能超过8单

                if action== 0 :

                    方向 = "多"

                    最大可买手数 = 保证金计算可买手数.calculate_tradable_lots(self.可用预存款 , self.杠杆倍数,
                                                                              self.摩擦成本, 上一步价格)
                    每份可支配手数 = np.floor((最大可买手数 / self.分仓数) * 100) / 100

                    数量 = 每份可支配手数
                    摩擦成本 = calculate_friction_cost(数量, self.杠杆倍数, self.摩擦成本, 开多考虑点差价格)

                    市值 = calculate_used_margin(float(数量), self.杠杆倍数, self.摩擦成本, 开多考虑点差价格)

                    #print(f"可用预存款弹夹规模预计算，free_margin{self.free_margin} ,每份可支配手数 {每份可支配手数}")


                    if self.可用预存款>市值:
                        #========================资金足够开仓======================================
                        self.可用预存款 -= 摩擦成本
                        self.可用预存款 -= 市值

                        operater(方向, 数量, 下单价=开多考虑点差价格, 摩擦成本=摩擦成本,
                                 止损百分比=self.止损百分比 / self.杠杆倍数, 市值=市值, 开仓市值=市值)
                        #print(f"已开多{当前子弹手数}手,摩擦成本{摩擦成本},市值{市值}")
                        self.上一个方向 = 方向
                        self.分仓列表.pop()

                        动作名称 = "开多" if action == 0 else "开空"

                        #print("###################################################################################################################")
                        #print(f"动作被触发！动作类型={动作名称}========================================================================================")

                        # 输出信息======================================================================================================
                        #print(f"Action_execution_countdown{self.Action_execution_countdown}")

                        #print(f"动作是：{动作名称}")
                        # #print(动作)
                        #print(f"--当前步已做决定步:{已做决定步}")
                        #print(f"-------操作后前持仓列表----变动前净值{总值}----")
                        #print(订单表.query_data())

                    else:
                        1
                        # #print(f'可用预存款不足！开单数量{数量},需要资金{市值+摩擦成本},可用资金{self.free_margin},当前净值{总值}')





                # 开空单的情况================================================================================================================
                if action ==1:
                    方向 = "空"

                    最大可买手数 = 保证金计算可买手数.calculate_tradable_lots(self.可用预存款, self.杠杆倍数,
                                                                              self.摩擦成本, 上一步价格)
                    每份可支配手数 = np.floor((最大可买手数 / self.分仓数) * 100) / 100

                    数量 = 每份可支配手数
                    摩擦成本 = calculate_friction_cost(数量, self.杠杆倍数, self.摩擦成本, 开多考虑点差价格)

                    市值 = calculate_used_margin(float(数量), self.杠杆倍数, self.摩擦成本, 开多考虑点差价格)

                    if self.可用预存款 > 市值:

                        self.可用预存款 -= 摩擦成本
                        self.可用预存款 -= 市值

                        operater(方向, 数量, 下单价=开多考虑点差价格, 摩擦成本=摩擦成本,
                                 止损百分比=self.止损百分比 / self.杠杆倍数, 市值=市值, 开仓市值=市值)
                        #print(f"已开空{当前子弹手数}手,摩擦成本{摩擦成本},市值{市值},当前止损")
                        self.上一个方向 = 方向
                        self.分仓列表.pop()

                        动作名称 = "开多" if action == 0 else "开空"

                        #print("###################################################################################################################")
                        #print(f"动作被触发！动作类型={动作名称}========================================================================================")

                        # 输出信息======================================================================================================
                        #print(f"Action_execution_countdown{self.Action_execution_countdown}")

                        #print(f"动作是：{动作名称}")
                        # #print(动作)
                        #print(f"--当前步已做决定步:{已做决定步}")
                        #print(f"-------操作后前持仓列表----变动前净值{总值}----")
                        #print(订单表.query_data())

                    else:
                        1
                        # #print(f'可用预存款不足！开单数量{数量},需要资金{市值+摩擦成本},可用资金{self.free_margin},当前净值{总值}')









        #更新净值+市值得到的总值==================================================================================
        订单表copy = 订单表.query_data()
        订单市值=订单表copy["市值"].sum()
        总值= self.可用预存款 + 订单市值




        # #print(f"账户净值{round(self.账户净值, 2)}")
        self.净值变化列表.append(总值)
        self.股价列表.append(self.close[未来的一步])
        # 计算奖励------------------------------------------------

        # 动作已经发生了，以上一系列计算也是对已经发生的动作的复盘，所以这里的奖励是对已经发生的动作的奖励
        '''
        在本步已经做完了订单流的操作，然后待定步(直到价格不知道怎么办)对本步做总结。
        从而知道本步做对没有，要求在当前计算完成后，
        也就是说一分钟刚好结束瞬间开始计算，一秒钟计算并下单，所有动作都在本步最后一秒钟完成，
        下一分钟刚好结束的时候，就对这一瞬间做汇总
        '''

        # reward = self._calculate_reward(action, 0)

        # 展望未来============================================================================
        # self.state = self._next_observation()

        # 环境到达终点===============================================================
        terminated = False
        truncated = False  # 根据需要设置
        if self.current_step >= len(self.data)-1 or 总值 <= 120:
            terminated = True
            truncated = False
            action_result = ActionResult(
                observation=1,
                reward=1,
                terminated=terminated,
                truncated=truncated,
                info={}
            )
            #=============像磁盘输出当前episode信息=======================
            # 保存=pd.DataFrame()
            # 保存["List_of_net_asset_value_changes"]=self.List_of_net_asset_value_changes
            # 保存["不计摩擦成本净值变化列表"]=self.不计摩擦成本净值变化列表
            # 保存["只计盈利的净值变化列表"]=self.只计盈利的净值变化列表
            # feather.write_dataframe(保存, r"C:\Users\joshua\Desktop\Pearl-main\pearl\外汇交易\episode")




        # 环境没有到达终点===============================================================
        else:
            # 创建并返回 ActionResult 对象===================================================
            action_result = ActionResult(
                observation=1,
                reward=1,
                terminated=terminated,
                truncated=truncated,
                info=总值
            )

            # #print(action_result)
        return action_result

    def reset(self, seed=None):

        # 重置环境
        self.current_step = 0
        self.state = self._next_observation()
        # 重置账户净值========================================================
        # ===========交易记录========================================================
        self.可用预存款 = 设定账户净值
        self.不计摩擦成本净值 = 设定账户净值
        self.只计盈利的净值 = 设定账户净值
        self.奖励累计=0
        self.延迟奖励累计=0
        self.奖励兑付倒计时=设定奖励兑付倒计时

        self.观察净盈亏 = 0
        self.观察摩擦成本 = 0
        self.分仓列表=[]
        self.净值变化列表 = []
        self.不计摩擦成本净值变化列表 = []
        self.只计盈利的净值变化列表 = []
        self.奖励累计列表 = []
        self.股价列表=[]
        self.股价净值差和=[]
        self.连续空仓时间=0

        self.分仓数=10

        self.上一个方向="多"
        self.相同方向计数=0

        self.子弹规模=0
        self.账户相对规模=0

        self.动作执行倒计时=action_countdown
        self.动作执行倒计时列表=[]
        self.上一个执行动作=0
        # 状态空间和动作空间初始化===========================================================

        operater("重置")
        return self.state, {}

    def map_value_to_range(self, value, ori_min,ori_max,new_min, new_max):
        """
        将一个在[-1, 1]范围内的值映射到新的给定范围[a, b]。

        参数:
        - value: 待映射的值，应在[-1, 1]范围内。
        - new_min: 新范围的最小值。
        - new_max: 新范围的最大值。

        返回:
        - 映射后的值。
        """
        # 原始范围的最小值和最大值
        original_min = ori_min
        original_max = ori_max

        # 映射公式
        mapped_value = ((value - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min

        return mapped_value

    def _calculate_reward(self, action, 上一个净值):


        return 1



    def _next_observation(self):

        return 1  # 转换为 NumPy 数组

    def render(self, mode='human'):
        if mode == 'human':
            if mode == 'human':
                with open('动态监控.json', 'w') as file:
                    json.dump(self.净值变化列表, file)
                with open('奖励动态监控.json', 'w') as file:
                    json.dump(self.奖励累计列表, file)




def 随机环境测试(a,b,c,d,e,f,回测开始百分比,回测结束百分比):
    indicator_column_names = ['二分高_Close_1', '二分高_Close_2', '二分高_Close_3', '二分高_Close_4', '二分高_Close_5',
                              '二分高_Close_6', '二分高_Close_7', '二分高_Close_8', '二分高_Close_9', '二分高_Close_10',
                              '二分高_Close_11', '二分高_Close_12', '二分高_Close_13', '二分高_Close_14',
                              '二分高_Close_15', '二分高_Close_16', '二分高_Close_17', '二分高_Close_18',
                              '二分高_Close_19', '二分高_Close_20', '二分低_Close_1', '二分低_Close_2',
                              '二分低_Close_3', '二分低_Close_4', '二分低_Close_5', '二分低_Close_6', '二分低_Close_7',
                              '二分低_Close_8', '二分低_Close_9', '二分低_Close_10', '二分低_Close_11',
                              '二分低_Close_12', '二分低_Close_13', '二分低_Close_14', '二分低_Close_15',
                              '二分低_Close_16', '二分低_Close_17', '二分低_Close_18', '二分低_Close_19',
                              '二分低_Close_20', '二分高simi_Close_1', '二分高simi_Close_2', '二分高simi_Close_3',
                              '二分高simi_Close_4', '二分高simi_Close_5', '二分高simi_Close_6', '二分高simi_Close_7',
                              '二分高simi_Close_8', '二分高simi_Close_9', '二分高simi_Close_10', '二分高simi_Close_11',
                              '二分高simi_Close_12', '二分高simi_Close_13', '二分高simi_Close_14',
                              '二分高simi_Close_15', '二分高simi_Close_16', '二分高simi_Close_17',
                              '二分高simi_Close_18', '二分高simi_Close_19', '二分高simi_Close_20', '二分低simi_Close_1',
                              '二分低simi_Close_2', '二分低simi_Close_3', '二分低simi_Close_4', '二分低simi_Close_5',
                              '二分低simi_Close_6', '二分低simi_Close_7', '二分低simi_Close_8', '二分低simi_Close_9',
                              '二分低simi_Close_10', '二分低simi_Close_11', '二分低simi_Close_12',
                              '二分低simi_Close_13', '二分低simi_Close_14', '二分低simi_Close_15',
                              '二分低simi_Close_16', '二分低simi_Close_17', '二分低simi_Close_18',
                              '二分低simi_Close_19', '二分低simi_Close_20']

    env = ForexTradingEnv(csv_file_path=r"C:\Users\joshua\Desktop\李卓宣框架\0新插值研究\重构后999.csv",
                          indicator_column_names=indicator_column_names,a=a,b=b,c=c,d=d,e=e,f=f,回测开始百分比=回测开始百分比,回测结束百分比=回测结束百分比)
    initial_state = env.reset()
    计数器=0
    while 1:
        sleep(0.005)
        action  = np.random.randint(0,2)
        env.render(mode='human')
        计数器+=1

        try:
            action_result = env.step(action).info
        except:
            return action_result

        if 计数器>1000 and action_result<7000:
            return action_result

        if 计数器>6000 and action_result<8500:
            return action_result


        # 输出交易后的状态、奖励和其他信息


if __name__ == '__main__':
    print(随机环境测试(a=-0.556884765625,b=-0.2001953125,c=0.973388671875,d=0.5,e=0.7,f=0.3,回测开始百分比=0,回测结束百分比=3.2))
