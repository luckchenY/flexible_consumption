import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial
from matplotlib.colors import ListedColormap
from scipy.stats import lognorm
from itertools import product
import random
import torch
from math import log
from math import exp
from envs.ev_trip_generator import EVTripGenerator
import json

class RouterRuleEnv(object):
    def __init__(self,
                    seed=42,
        ):
        super(RouterRuleEnv, self).__init__()
        # 使用Python默认的float64类型
        self.max_ES_capacity = 1200.0  # 300kwh 1200kw`15min
        self.ES_capacity = self.max_ES_capacity / 2
        self.EV_com = []
        self.EV_info = []
        self.PV_GEN = []
        self.WT_GEN = []
        self.PV_F = []
        self.WT_F = []
        self.household_ele = []
        self.fundamental = [0.0] * 50  # 每辆车的实时充电量数组
        
        self.ES_ratedcharge = 100.0  # kw
        self.ES_rateddischare = 100.0  # kw
        self.PG_rated = 100.0  # kw
        self.maxEVpower = 60.0  # kw
        self.PVprice = 0.001  # PV价格 kw15min
        self.WTprice = 0.001  # WT价格 kw15min
        self.pricepuchase = []
        self.household_price = []
        self.state_dim = 5  # (sum p^f_t, forcast PT, forcast WT, es_capacity, t )
        self.seed = seed
        self.eleconsumption_km = 0  # kW15min/km
    
        self.EVbattery = 240 # 60kwh 240kw`15min
        self.available_EV = 0
        # 电池成本系数 (Battery cost coefficients)
        self.batterdy_deg=  0.00425 #每kw 15mmin的电池损耗成本 
        #光伏一天800左右, 储能300kwh, 电动汽车100kwh* 50=5000kwh, 家庭10kwh *100 =1000kwh,
        self.profit= 1.2
    def reset(self, season):
        # 记录本轮次(episode)中的power_input总和
        self.episode_power_input_sum = 0.0
        self.soc = 0.4
        
        # 根据季节选择家庭用电数据
        if season == 'winter':
            df = pd.read_csv('envdata/100houseloadwinter.txt', sep=';', header=None)
        elif season == 'summer':
            df = pd.read_csv('envdata/100houseloadsummer.txt', sep=';', header=None)
        else:
            raise ValueError(f"无效的季节参数: {season}。只支持 'winter' 或 'summer'")
        
        # 将数据转换为float64类型
        self.household_ele = (df[1] / 1000.0).tolist()  # 一天是1000多，小汽车一天60 * 50 *0.2=600
        # print("household_ele", sum(self.household_ele[0:96]))  # 打印前10个数据进行检查
       
        # 根据季节选择发电数据文件
        if season == 'winter':
            actual_gen_file = 'envdata/Actual_generation_winter.csv'
            forecast_file = 'envdata/Generation_Forecast_Intraday_winter.csv'
            price_file='envdata/winterprice.csv'
            self.eleconsumption_km = 1  # kW15min/km
        elif season == 'summer':
            actual_gen_file = 'envdata/Actual_generation_summer.csv'
            forecast_file = 'envdata/Generation_Forecast_Intraday_summer.csv'
            price_file='envdata/summerprice.csv'
            self.eleconsumption_km = 0.8  # kW15min/km
        else:
            raise ValueError(f"无效的季节参数: {season}。只支持 'winter' 或 'summer'")
        
        # 加载实际发电数据
        try:
            df = pd.read_csv(actual_gen_file, sep=';', thousands=',')
            # 确保数据类型为数值型
            self.PV_GEN = (pd.to_numeric(df['Photovoltaics [MWh] Original resolutions'])/10).astype(float).tolist()  # Photovoltaics column (index 4)
            self.WT_GEN = (pd.to_numeric(df['Wind onshore [MWh] Original resolutions'])/10).astype(float).tolist()  # Wind onshore column (index 3)
            #接近一千的光伏峰值，
            # print("PV",sum(self.PV_GEN[0:960]))  
            # print("WT",sum(self.WT_GEN[0:960]))  
        except FileNotFoundError:
            print(f"警告: 找不到文件 {actual_gen_file}")
       
        # 加载预测发电数据
        try:
            df = pd.read_csv(forecast_file, sep=';', thousands=',')
            # 确保数据类型为数值型
            self.PV_F = (pd.to_numeric(df['Photovoltaics [MWh] Original resolutions'])/10).astype(float).tolist()  # Photovoltaics column (index 4)
            self.WT_F = (pd.to_numeric(df['Wind onshore [MWh] Original resolutions'])/10).astype(float).tolist()  # Wind onshore column (index 3)
            
        except FileNotFoundError:
            print(f"警告: 找不到文件 {forecast_file}")
           
        # 加载价格数据
        try:
            df = pd.read_csv(price_file,header=None) # 替换为你的文件名
            self.pricepuchase = pd.to_numeric(df.iloc[:, 0]).astype(float).tolist()

            # # 加载价格数据
            # df = pd.read_excel('envdata/Price_EVLink.xlsx')  # 替换为你的文件名
            # price_list = (df['PRICE'] / 4.0).tolist()
            # # print(len(price_list))
            # # 先把每个元素复制3次
            # price_list = np.repeat(price_list, 2).tolist()
            # # 然后把整个列表复制7次
            # self.pricepuchase = np.tile(price_list, 7).tolist()
            # print(len(self.pricepuchase))

            # print(self.pricepuchase.iloc[0:50])  
        except FileNotFoundError:
            print(f"警告: 找不到文件 {price_file}")
        
        self.ES_capacity = self.max_ES_capacity / 2
        
        self.available_EV = 0
        
        # EV_com: 从JSON加载原始 trips_data；失败则现场生成，然后按“返回时刻”分组
        try:
            with open('envdata/ev_trips_14days.json', 'r', encoding='utf-8') as f:
                trips_data = json.load(f)
        except Exception:
            print("Failed to load trips_data from JSON file, generating new trips_data")
            trips_data = EVTripGenerator().generate_all_cars_trips(days=14)
        self.EV_com = self.build_ev_return_groups(trips_data)
        # EV_info: 固定50个车辆的状态 [PARKING_DURATION_slots, NEXT_MILEAGE_km, BATTERY, can_charge]
        self.EV_info = [[0, 0.0, 0.4 * self.EVbattery, False] for _ in range(50)]
        
        state_init = [0.0] * self.state_dim
        state_init[0] = sum(self.fundamental) 
        state_init[1] = self.PV_F[0]
        state_init[2] = self.WT_F[0]
        state_init[3] = self.ES_capacity
        state_init[4] = 0.0
        return state_init
        
    def build_ev_return_groups(self, trips_data):
        """遍历所有车辆的旅行数据，按“回来时间(全局timeslot)”分组，
        记录 (car_id, next_trip_distance_km, dwell_time_slots)。
        dwell_time = 下一次出发时刻 - 本次返回时刻。若无下一次出发则跳过。
        返回: dict[int, list[tuple[int, float, int]]]
        """
        groups = {}
        for car_id, trips in enumerate(trips_data):
            
            for idx, trip in enumerate(trips):
                if not (isinstance(trip, list) and len(trip) >= 3):
                    continue
                arrive_slot = int(trip[2])
                # 找下一次出发
                if idx + 1 < len(trips):
                    next_trip = trips[idx + 1]
                    if not (isinstance(next_trip, list) and len(next_trip) >= 2):
                        continue
                    next_start = int(next_trip[0])
                    next_distance = float(next_trip[1])
                    dwell_slots = max(0, next_start - arrive_slot)
                    assert dwell_slots > 0
                    groups.setdefault(arrive_slot, []).append((car_id, next_distance, dwell_slots))       
                else:
                    next_start = arrive_slot + 96
                    next_distance = trips[-1][1] #和这次的保持一样
                    dwell_slots = 96
                    groups.setdefault(arrive_slot, []).append((car_id, next_distance, dwell_slots))       
        return groups

    def process_ev_arrivals_to_info(self, time_slot: int):
        """针对当前全局 timeslot 的到达车辆，用 EV_com 分组信息更新 EV_info。
        EV_com 为 dict[到达时刻] -> list[(car_id, next_distance_km, dwell_slots)]。
        EV_info[car_id] = [dwell_slots, next_distance_km, battery_units, can_charge]
        """
        # 重新计算本时隙的 fundamental，避免累计叠加
        self.fundamental = [0.0] * 50  # 重置所有车辆的充电量
        arrivals = self.EV_com.get(int(time_slot), []) if isinstance(self.EV_com, dict) else []
        for car_id, next_distance, dwell_slots in arrivals:
            # 检查车辆是否已经在站内
            if self.EV_info[car_id][3]:  # 已经在站内
                continue
            
            # 检查是否需要充电
            if self.EV_info[car_id][2] >= 0.7*self.EVbattery:
                continue
            
            if dwell_slots <5:
                continue
            # 更新车辆状态
            self.EV_info[car_id] = [int(dwell_slots), float(next_distance), self.EV_info[car_id][2], True]
            self.available_EV += 1
        
        for i in range (len(self.EV_info)):
            if self.EV_info[i][3]:
                self.fundamental[i] = max(0, self.EV_info[i][1] * self.eleconsumption_km-self.EV_info[i][2]) / self.EV_info[i][0]
                
    def calculate_battery_cost(self, power):
        battery_cost = self.batterdy_deg*power
        return battery_cost
    def calculate_powerpurchasefee(self,power,t):
        power_cost= (self.pricepuchase[t]+0.03274)*power/4
        return power_cost

    def log_curve_mapping(self, x):
        """
        将输入 x 从 [-1, 1] 映射到 [0, self.PG_rated]，使用log曲线
        """
        # 将[-1, 1]映射到[0, 1]
        x_normalized = (x + 1) / 2  # x从[-1,1]映射到[0,1]
        # log 映射，避免 log(0)
        log_input = x_normalized * 9.0 + 1.0  # log_input在[1, 10]范围内
        # print("log_input", log_input)
        log_x = np.log(log_input) / np.log(10.0)  # log 曲线，0~1 映射到 0~1
        
        mapped = log_x * self.PG_rated
        
        return mapped

    def clip_action (self, consumptionflag, action,t):
        action = action.astype(float)
        assert self.ES_capacity >= 0
        assert self.ES_capacity <= self.max_ES_capacity
        assert sum(self.fundamental) >= 0

        #power_input 给用户端最终的电， PGS 电网给储能， PG 电网给用户， power_discharge, power_charge
        if consumptionflag > 0:

            action[0] = (action[0] + 1) / 2 * self.ES_rateddischare  # 电池放电功率
            ESSoutputcapacity= min(self.ES_capacity - 0.1*self.max_ES_capacity, self.ES_rateddischare)
            
            if ESSoutputcapacity < action[0] :
                action[0] = ESSoutputcapacity

            action[1] = self.log_curve_mapping(action[1])  # 电网功率

            PG=action[1]
            power_discharge=action[0]

            demand = self.household_ele[t]+sum(self.fundamental)
            power_input = self.PV_GEN[t] + self.WT_GEN[t] +PG + power_discharge
            
            if power_input < demand: 
                action[1] = demand - (self.PV_GEN[t] + self.WT_GEN[t] + power_discharge)
                power_input = demand
                PG=action[1]           
                
            if power_input > self.household_ele[t] + self.maxEVpower*self.available_EV:
                #超过消纳能力，惩罚奖励
                PG = self.household_ele[t] + self.maxEVpower*self.available_EV - (self.PV_GEN[t] + self.WT_GEN[t] + power_discharge)
                
                if PG < 0:
                    action[1]-=PG
                    PG=0
                    power_discharge=action[0]
                power_input= self.household_ele[t] + self.maxEVpower*self.available_EV
               
            return PG, power_input, power_discharge

        else:
            if self.PV_F[t] + self.WT_F[t] >= self.household_ele[t] + self.maxEVpower * self.available_EV + max(0,0.9*self.max_ES_capacity - self.ES_capacity):
                print("必须要发生弃电",t%96)
                power_storage=0.9*self.max_ES_capacity - self.ES_capacity
                power_input=self.household_ele[t] + self.maxEVpower * self.available_EV
                PG=0
                return power_input, power_storage, PG

            action[0] = (action[0] + 1) / 2
            PVS = self.PV_GEN[t] * action[0]  # 储存的PV功率
            WTS = self.WT_GEN[t] * action[0]
            demand = self.household_ele[t] + sum(self.fundamental)

            action[1] = self.log_curve_mapping(action[1])  #电网向储能
            action[2] = self.log_curve_mapping(action[2])  #电网向用户 
            power_storage = PVS + WTS + action[1]
            power_input = self.PV_GEN[t] - PVS+ self.WT_GEN[t]-WTS+action[2]
            PG = action[1] +action[2]  # 电网    

            storage_capability=min(0.9*self.max_ES_capacity-self.ES_capacity,self.ES_ratedcharge)
            assert storage_capability >= 0
            if power_storage > storage_capability:
                action[1] = 0.0
                power_storage = PVS + WTS 
                PG= action[2]  # 电网功率

                if power_storage > storage_capability:  #需要减少储能比例
                    power_storage = storage_capability
                    
                    power_input = self.PV_GEN[t]+ self.WT_GEN[t] - storage_capability +action[2]

            if power_input < demand:
                action[2] +=(demand-power_input)
                PG=action[2]+action[1]
                power_input = demand
            
            maxdemand=self.household_ele[t] + self.maxEVpower*self.available_EV
            if power_input > maxdemand:
                power_input = maxdemand
                action[2]-= (maxdemand-power_input)
                #如果新能源发电的综合大于消纳能力，会直接进行消纳。也就是说acion[2]
                assert action[2] >= 0   
                PG = action[2] + action[1]
                
            return power_input, power_storage, PG

    def env_knowledge(self,t,season):

        #太多了必须进行储能
        if self.PV_F[t] + self.WT_F[t] >= self.household_ele[t] + self.maxEVpower * self.available_EV: #消纳端放不下，必须进行储能,这里有可能弃电
            return 1
        elif self.PV_F[t]+ self.WT_F[t] +self.PG_rated <= self.household_ele[t] + sum(self.fundamental): #需要电池的电才够用
            return 2
        
        if self.ES_capacity >= 0.9*self.max_ES_capacity:
            return 2 #放电
        elif self.ES_capacity <= 0.1*self.max_ES_capacity:  # 如果储能电池电量为0
            return 1 #储能
        
        #设置一些什么时候充放电的规则
        if season=='summer':
            if self.pricepuchase[t]< 0.01:
                return 1
            else:
                return 2
        else:
            if self.pricepuchase[t]< 0.03:
                return 1
            else:
                return 2


    def step(self, t, consumptionflag, RL_action):
        #完成状态转移
    
        #如果没有EV在的话，也就没有办法进行消纳,用户侧是固定的，如果电池放电，就是新能源 +电池放电+ 电网送电， 和是居民用电。如果电池充电，就是新能源+电网， 和是居民用电。
        active_cars = [i for i in range(50) if self.EV_info[i][3]]
        if len(active_cars) == 0:
            if self.PV_GEN[t] + self.WT_GEN[t] >= self.household_ele[t]:
                # 计算充电功率和电池成本
                power_storage = min(self.PV_GEN[t]+self.WT_GEN[t] - self.household_ele[t], self.ES_ratedcharge)
                battery_cost = self.calculate_battery_cost(power_storage)
                PG=0
                reward = - battery_cost - self.PV_GEN[t] * self.PVprice - self.WT_GEN[t] * self.WTprice
                self.ES_capacity = min(0.9*self.max_ES_capacity, self.ES_capacity + power_storage)
            else:
                power_discharge = self.household_ele[t] - self.PV_GEN[t]-self.WT_GEN[t]
                PG=0
                if power_discharge > self.ES_capacity - 0.1*self.max_ES_capacity:
                    power_discharge = self.ES_capacity - 0.1*self.max_ES_capacity
                    PG = self.household_ele[t] - self.PV_GEN[t]-self.WT_GEN[t]-power_discharge
                self.ES_capacity = self.ES_capacity - power_discharge

                battery_cost = self.calculate_battery_cost(power_discharge)
                PG_cost=self.calculate_powerpurchasefee(PG,t)
                reward= -PG_cost-battery_cost - self.PV_GEN[t] * self.PVprice - self.WT_GEN[t] * self.WTprice
            
            power_input = self.household_ele[t]

        else:
            
            if consumptionflag==1: 
                assert self.ES_capacity >= 0.1*self.max_ES_capacity
                PG, power_input, power_discharge = self.clip_action(consumptionflag, RL_action,t)
                assert PG >= 0
                assert power_discharge >= 0
                assert power_input>=self.household_ele[t] + sum(self.fundamental)
                assert power_input<=self.household_ele[t] + self.maxEVpower*self.available_EV
                # 只对在站内且可充电的车辆分配充电功率
                consumptionpower = (power_input - self.household_ele[t] - sum(self.fundamental)) / self.available_EV
                for i in active_cars:
                    self.EV_info[i][2] = self.EV_info[i][2] + self.fundamental[i] + consumptionpower
                    self.EV_info[i][0] = self.EV_info[i][0] - 1
                    
                # print("ES_capacity, power_discharge", self.ES_capacity, power_discharge)
                self.ES_capacity = self.ES_capacity - power_discharge 
                
                # 计算电池放电成本
                battery_cost = self.calculate_battery_cost(power_discharge)
                PG_cost=self.calculate_powerpurchasefee(PG,t)
                reward = - PG_cost - battery_cost - self.PV_GEN[t] * self.PVprice - self.WT_GEN[t] * self.WTprice
            else:
                assert self.ES_capacity <= 0.9*self.max_ES_capacity
                power_input, power_storage, PG = self.clip_action(consumptionflag,RL_action,t)
                # 只对在站内且可充电的车辆分配充电功率
                assert PG >= 0
                assert power_storage >= 0
                assert power_input>=self.household_ele[t] + sum(self.fundamental)
                assert power_input<=self.household_ele[t] + self.maxEVpower*self.available_EV
                consumptionpower = (power_input - self.household_ele[t] - sum(self.fundamental)) / len(active_cars)
                for i in active_cars:
                    self.EV_info[i][2] = self.EV_info[i][2] + self.fundamental[i] + consumptionpower
                    self.EV_info[i][0] = self.EV_info[i][0] - 1
                    
                self.ES_capacity = self.ES_capacity + power_storage
                
                # 计算电池充电成本
                battery_cost = self.calculate_battery_cost(power_storage)
                PG_cost=self.calculate_powerpurchasefee(PG,t)   
                reward = - battery_cost - PG_cost - self.PV_GEN[t] * self.PVprice - self.WT_GEN[t] * self.WTprice

        # 累加本episode的power_input
        self.episode_power_input_sum += power_input

        # 电动汽车充完电的清除
        for i in range(50):
            # 元素: [duration, mileage_km, battery_loc, can_charge]
            if self.EV_info[i][3]:
                duration_slots = self.EV_info[i][0]
                battery_units = self.EV_info[i][2]
                
                if duration_slots == 0 or battery_units >= 0.95*self.EVbattery:  # 停车结束或已充满就清楚这辆车
                    self.EV_info[i][3] = False
                    self.EV_info[i][2] -= self.EV_info[i][1] * self.eleconsumption_km #更新为下一次来充电的时候的电量
                    self.available_EV -= 1
                
    
        self.process_ev_arrivals_to_info(t+1)

        next_state = [0.0] * 5

        assert self.ES_capacity <=self.max_ES_capacity
        assert self.ES_capacity >= 0
        assert sum(self.fundamental)>=0
        assert self.available_EV>=0
        assert self.available_EV <= 50

        next_state[0]= sum(self.fundamental)  
        next_state[1]= self.PV_F[t+1]
        next_state[2]= self.WT_F[t+1]
        next_state[3]= self.ES_capacity
        next_state[4]= t%96
        # print("next_state", next_state)

        if PG>self.household_ele[t]+sum(self.fundamental):
            price_f= self.profit*self.pricepuchase[t]
            PGpropotion=(PG-self.household_ele[t]+sum(self.fundamental)) / ((PG-self.household_ele[t]+sum(self.fundamental))+self.PV_GEN[t]+self.WT_GEN[t])
            price_c= self.profit* (self.pricepuchase[t]*PGpropotion + (1-PGpropotion)* self.PVprice)
        else:
            price_c=self.profit*self.PVprice
            PGpropotion=PG/(self.household_ele[t]+sum(self.fundamental))
            price_f=self.profit* (self.pricepuchase[t]*(PGpropotion) + (1-PGpropotion)* self.PVprice )
        energyfee= price_c* (power_input-self.household_ele[t]+sum(self.fundamental))+ price_f * (self.household_ele[t]+sum(self.fundamental)) 
        purchase_fee= self.calculate_powerpurchasefee(PG,t)
        done=False
        
        return next_state, reward, done ,price_f, price_c,purchase_fee,energyfee