
import datetime
import os
import pickle
import geatpy as ea  # import geatpy
import numpy as np
import pandas as pd

from pearl.外汇交易分仓.随机环境宽松 import 随机环境测试
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)  # 对于行
pd.set_option('display.max_columns', None)  # 对于列
pd.set_option('display.width', None)  # 调整显示宽度，以便完整显示每行
pd.set_option('display.max_colwidth', None)  # 对于每个列值的宽度

class soea_multi_SEGA_joshua(ea.SoeaAlgorithm):
    """
soea_multi_SEGA_templet : class - Multi-population Strengthen Elitist GA Algorithm(增强精英保留的多种群协同遗传算法类)

算法类说明:
    该算法类是内置算法类soea_SEGA_templet的多种群协同版本，为不同的种群设置不同的重组和变异概率。
    注意：本算法类中的population为一个存储种群类对象的列表，而不是单个种群类对象。

算法描述:
    本算法类实现的是增强精英保留的多种群协同遗传算法。算法流程如下：
    1) 循环population列表，初始化列表中的各个种群的染色体，并将所有种群所有个体的数目记录为NindAll。
    2) 若满足进化算法停止条件则停止，否则继续执行。
    3) 循环对各个种群独立进行选择、重组、变异，得到各个种群的子代，并将父代和子代个体合并。
    4) 对所有种群的所有个体进行统一的适应度评价。
    5) 根据适应度调用选择算子进行环境选择，选择出NindAll个个体形成新一代种群。
    6) 根据概率进行种群间个体迁移。
    7) 回到第2步。

"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         trappedValue, maxTrappedCount, dirName)
        if type(population) != list:
            raise RuntimeError('传入的种群对象列表必须为list类型')

        self.name = 'multi-SEGA'
        self.PopNum = len(population)  # 种群数目
        self.selFunc = 'otos'  # 锦标赛选择算子
        self.migFr = 5  # 发生种群迁移的间隔代数
        self.是否第一次运行=1
        self.migOpers = ea.Migrate(MIGR=0.2, Structure=2, Select=1, Replacement=2)  # 生成种群迁移算子对象
        # 为不同的种群设置不同的重组、变异算子
        self.recOpers = []
        self.mutOpers = []
        Pms = np.linspace(1 / self.problem.Dim, 1, self.PopNum)  # 生成变异概率列表，为不同的种群分配不同的变异概率
        Pcs = np.linspace(0.7, 1, self.PopNum)  # 生成重组概率列表，为不同的种群分配不同的重组概率
        for i in range(self.PopNum):  # 遍历种群列表
            pop = population[i]  # 得到当前种群对象
            if pop.Encoding == 'P':
                recOper = ea.Xovpmx(XOVR=Pcs[i])  # 生成部分匹配交叉算子对象
                mutOper = ea.Mutinv(Pm=float(Pms[i]))  # 生成逆转变异算子对象
            else:
                recOper = ea.Xovdp(XOVR=Pcs[i])  # 生成两点交叉算子对象
                if pop.Encoding == 'BG':
                    mutOper = ea.Mutbin(Pm=float(Pms[i]))  # 生成二进制变异算子对象
                elif pop.Encoding == 'RI':
                    mutOper = ea.Mutbga(Pm=float(Pms[i]), MutShrink=0.5, Gradient=20)  # 生成breeder GA变异算子对象
                else:
                    raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
            self.recOpers.append(recOper)
            self.mutOpers.append(mutOper)

    def unite(self, population):
        """
        合并种群，生成联合种群。
        注：返回的unitePop不携带Field和Chrom的信息，因为其Encoding=None。
        """




        # 遍历种群列表，构造联合种群
        unitePop = ea.Population(None, None, population[0].sizes, None,  # 第一个输入参数传入None，设置Encoding为None
                                 ObjV=population[0].ObjV,
                                 FitnV=population[0].FitnV,
                                 CV=population[0].CV,
                                 Phen=population[0].Phen)

        for i in range(1, self.PopNum):
            unitePop += population[i]
        return unitePop

    def calFitness(self, population):
        """
        计算种群个体适应度，population为种群列表
        该函数直接对输入参数population中的适应度信息进行修改，因此函数不用返回任何参数。
        """
        ObjV = np.vstack(list(pop.ObjV for pop in population))
        CV = np.vstack(list(pop.CV for pop in population)) if population[0].CV is not None else population[0].CV
        FitnV = ea.scaling(ObjV, CV, self.problem.maxormins)  # 统一计算适应度
        # 为各个种群分配适应度
        idx = 0
        for i in range(self.PopNum):
            population[i].FitnV = FitnV[idx: idx + population[i].sizes]
            idx += population[i].sizes

    def EnvSelection(self, population, NUM):  # 环境选择，选择个体保留到下一代

        FitnVs = list(pop.FitnV for pop in population)



        NewChrIxs = ea.mselecting('dup', FitnVs, NUM)  # 采用基于适应度排序的直接复制选择
        for i in range(self.PopNum):
            population[i] = (population[i])[NewChrIxs[i]]

        #===================================每一步保存模型=============================================================
        if self.是否第一次运行==0:
            file_name = '保存模型.pkl'
            # 使用pickle将对象保存到文件
            with open(file_name, 'wb') as file:
                pickle.dump(population, file)
        #===================================每一步保存模型=============================================================
        return population

    def run(self, prophetPops=None):  # prophetPops为先知种群列表（即包含先验知识的种群列表）
        # ==========================初始化配置===========================
        self.initialization()  # 初始化算法类的一些动态参数
        population = self.population  # 密切注意本算法类的population是一个存储种群类对象的列表

        #============第一次运行加载旧模型====================================================================================
        if self.是否第一次运行==1:
            # 指定要加载的文件名
            file_name = '保存模型.pkl'
            # 检查文件是否存在
            if os.path.exists(file_name):
                # 使用pickle从文件中加载对象
                with open(file_name, 'rb') as file:
                    population = pickle.load(file)
                print("加载到旧模型")
            else:
                print(f"The file '{file_name}' does not exist.")
            self.是否第一次运行 = 0
        #============第一次运行加载旧模型====================================================================================


        try:
            NindAll = 0  # 记录所有种群个体总数
            # ===========================准备进化============================
            for i in range(self.PopNum):  # 遍历每个种群，初始化每个种群的染色体矩阵
                NindAll += population[i].sizes
                # population[i].initChrom(population[i].sizes)  # 初始化种群染色体矩阵
                # 插入先验知识（注意：这里不会对先知种群列表prophetPops的合法性进行检查）
                if prophetPops is not None:
                    population[i] = (prophetPops[i] + population[i])[:population[i].sizes]  # 插入先知种群
                self.call_aimFunc(population[i])  # 计算种群的目标函数值
        except:
            NindAll = 0  # 记录所有种群个体总数
            # ===========================准备进化============================
            for i in range(self.PopNum):  # 遍历每个种群，初始化每个种群的染色体矩阵
                NindAll += population[i].sizes
                population[i].initChrom(population[i].sizes)  # 初始化种群染色体矩阵
                # 插入先验知识（注意：这里不会对先知种群列表prophetPops的合法性进行检查）
                if prophetPops is not None:
                    population[i] = (prophetPops[i] + population[i])[:population[i].sizes]  # 插入先知种群
                self.call_aimFunc(population[i])  # 计算种群的目标函数值



        self.calFitness(population)  # 统一计算适应度
        unitePop = self.unite(population)  # 得到联合种群unitePop
        # ===========================开始进化============================
        while self.terminated(unitePop) == False:
            for i in range(self.PopNum):  # 遍历种群列表，分别对各个种群进行重组和变异


                pop = population[i]  # 得到当前种群
                # 选择
                offspring = pop[ea.selecting(self.selFunc, pop.FitnV, pop.sizes)]
                # 进行进化操作
                offspring.Chrom = self.recOpers[i].do(offspring.Chrom)  # 重组
                offspring.Chrom = self.mutOpers[i].do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
                self.call_aimFunc(offspring)  # 计算目标函数值
                population[i] = population[i] + offspring  # 父子合并
            self.calFitness(population)  # 统一计算适应度
            population = self.EnvSelection(population, NUM=NindAll)  # 选择个体得到新一代种群
            if self.currentGen % self.migFr == 0:
                population = self.migOpers.do(population)  # 进行种群迁移


            unitePop = self.unite(population)  # 更新联合种群
        return self.finishing(unitePop)  # 调用finishing完成后续工作并返回结果


def get_key_from_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None  # 如果没有找到匹配的键，返回None或其他适当的值


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'
        M = 1
        maxormins = [-1]

        #========================设置具体参数取值范围========================================

        # <editor-fold desc="设置所有参数的取值范围">
        #正向参数[权重, 采样目标类型, 采样周期, 多少日窗口, 多大插值, 目标股票, 目标周期, 采样窗口点1, 采样窗口点2]

        #每次运行的时候要手动删除保存的模型，否则就会遇到参数超出取值范围的情况
        参数a=[-2,2]
        参数b=[-2,2]
        参数c=[-2,2]
        参数d=[-2,2]
        参数e=[-2,2]
        参数f=[-2,2]


        # </editor-fold>

        参数汇总 = [参数a,参数b,参数c,参数d,参数e,参数f]
        # print(f"参数数量是{len(参数汇总)}")
        # <editor-fold desc="自动转置参数范围">

        #====================把这些参数的取值范围从[[小,大]，[小,大]]改成[[小小小小],[大大大大]]
        所有参数最小取值范围=np.array(参数汇总).transpose()[0]
        所有参数最大取值范围=np.array(参数汇总).transpose()[1]

        参数数量=len(参数汇总)
        Dim = 参数数量  # 设置所有参数的数量
        varTypes = [0] * Dim
        lb = 所有参数最小取值范围  # 设置所有参数取值范围最小
        ub = 所有参数最大取值范围   # 设置所有参数取值范围最大
        lbin = [1] * Dim
        ubin = [1] * Dim
        # </editor-fold>
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, x):
        长度=len(x)
        # print("===========================开始观察了=======================================================")
        # print(x)
        # print("==========================================================================================")

        # <editor-fold desc="按顺序初步解包所有的内容">

        参数a包 = x[:, [0]]
        参数b包 = x[:, [1]]
        参数c包 = x[:, [2]]
        参数d包 = x[:, [3]]
        参数e包 = x[:, [4]]
        参数f包 = x[:, [5]]


        种群全部结果=[]
        for i in range(长度): #这里把[[1],[2]....]这种数据取出来


            # 正向参数[权重, 采样目标类型, 采样周期, 多少日窗口, 多大插值, 目标股票, 目标周期, 采样窗口点1, 采样窗口点2]

            参数a = 参数a包[i][0]
            参数b =参数b包[i][0]
            参数c = 参数c包[i][0]
            参数d = 参数d包[i][0]
            参数e =参数e包[i][0]
            参数f = 参数f包[i][0]

            # 正向参数[权重, 采样目标类型, 采样周期, 多少日窗口, 多大插值, 目标股票, 目标周期, 采样窗口点1, 采样窗口点2]
            适应度列表=[]
            for i in range(1):
                适应度=随机环境测试(a=参数a, b=参数b, c=参数c,d=参数d,e=参数e,f=参数f, 回测开始百分比=0, 回测结束百分比=60)

                # print(f"第{i+1}次尝试 参数a是:{参数a} 参数b是:{参数b} 参数c是: {参数c} 适应度{适应度}")
                # if 适应度<9000:
                #     适应度列表.append(适应度*0.1)
                #     print("适应度过低！小于9000！提前停止")
                #     break
                # else:
                #     适应度列表.append(适应度)

            # 平均适应度=np.mean(适应度列表)
            平均适应度=适应度

            #保存参数记录功能===================================================================================
            # 路径指向您想要检查或存储的dataframe文件
            df_path = '参数记录.csv'
            # 检查文件是否存在
            if os.path.isfile(df_path):
                # 读取现有dataframe
                df = pd.read_csv(df_path)
            else:
                # 创建新的dataframe
                df = pd.DataFrame(columns=['参数a', '参数b', '参数c','参数d', '参数e', '参数f', '平均适应度'])
            # 假设这里有一行新数据，您需要替换下面的new_data为实际数据

            print(f"尝试完成！参数a是:{参数a} 参数b是:{参数b} 参数c是: {参数c} 平均适应度{平均适应度}")

            new_data = pd.DataFrame({'参数a': [参数a], '参数b': [参数b], '参数c': [参数c],'参数d': [参数d], '参数e': [参数e], '参数f': [参数f], '平均适应度': [平均适应度]})

            # 添加新数据到dataframe中
            df = pd.concat([df, new_data], ignore_index=True)
            # 保存dataframe到文件
            df = df.sort_values(by='平均适应度', ascending=False)
            df=df.reset_index(drop=True)
            df.to_csv(df_path, index=False)
            # 显示更新后的dataframe内容
            print(df[:10])

            f = 平均适应度
            #传入适应度函数===============================================
            种群全部结果.append([f])

        # 转置数组，使其变成列向量的形式
        种群全部结果 = np.array(种群全部结果)
        # print(种群全部结果)


        return 种群全部结果



if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 种群设置
    Encoding = 'RI'  # 编码方式
    NINDs = [5, 10, 15, 20]  # 种群规模
    population = []  # 创建种群列表
    for i in range(len(NINDs)):
        Field = ea.crtfld(Encoding,
                          problem.varTypes,
                          problem.ranges,
                          problem.borders)  # 创建区域描述器。
        population.append(ea.Population(
            Encoding, Field, NINDs[i]))  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）。
    # 构建算法
    algorithm = soea_multi_SEGA_joshua(
        problem,
        population,
        MAXGEN=30,  # 最大进化代数。
        logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
        trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。 准确率进步＜1的-6次方就陷入停滞
        maxTrappedCount=20)  # 进化停滞计数器最大上限值。 停滞多少次就停止优化
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=False)
    print(res)