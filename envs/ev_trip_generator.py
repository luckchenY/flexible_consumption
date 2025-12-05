import numpy as np
import random
from datetime import datetime, timedelta
import json
import math

class EVTripGenerator:
    def __init__(self):
        self.num_cars = 50
        self.commuter_cars = 20  # 早出晚归模式
        self.short_trip_cars = 30  # 短距离出行模式
        
        # 时间参数（小时）
        self.morning_start = 6
        self.morning_end = 9
        self.evening_start = 17
        self.evening_end = 22
       
        self.afternoon_start = 14
        self.afternoon_end = 16
        
        # 距离参数（公里）
        self.commuter_distance_mean = math.log(50)  # 对数正态分布均值
        self.commuter_distance_sigma = 1
        self.short_trip_distance_mean = math.log(20)
        self.short_trip_distance_sigma = 1
        
        # 行驶速度（km/h）
        self.avg_speed = 50
        
    def generate_lognormal_distance(self, mean, sigma):
        """生成对数正态分布的距离"""
        return round(max(0.5, np.random.lognormal(mean, sigma)), 2)
    
    def calculate_trip_duration(self, distance):
        """根据距离计算行驶时间（小时）"""
        return round(distance / self.avg_speed, 2)
    
    def is_time_conflict(self, new_start, new_end, existing_trips):
        """检查时间冲突（existing_trips 里时间可能已是slot，需还原为小时制）"""
        for trip in existing_trips:
            existing_start = trip[0]
            existing_end = trip[2]
            # 若是离散slot（int），还原为小时
            if isinstance(existing_start, (int, np.integer)):
                existing_start = existing_start / 4.0
            if isinstance(existing_end, (int, np.integer)):
                existing_end = existing_end / 4.0
            # True 表示有重叠
            if not (new_end + 0.25 <= existing_start or new_start >= existing_end + 0.25):
                return True
        return False
    
    def ensure_time_in_range(self, time_value):
        """确保时间在0-24小时内（支持标量或numpy数组）"""
        if isinstance(time_value, np.ndarray):
            return np.clip(time_value, 0, 24)
        return max(0, min(time_value, 24))

    def to_slot(self, time_value):
        """将小时制时间离散为15分钟时段索引（0-95），支持标量或numpy数组。"""
        t = self.ensure_time_in_range(time_value)
        if isinstance(t, np.ndarray):
            slots = (t * 4).astype(int)
            return np.clip(slots, 0, 95)
        slot = int(t * 4)
        return 95 if slot > 95 else slot

    def generate_commuter_trip_single(self, day_index=0):
        """生成单辆车的一次通勤出行（早出晚归），按天偏移slot。"""
        # 参数
        morning_mean = (self.morning_start + self.morning_end) / 2
        morning_std = (self.morning_end - self.morning_start) / 6
        evening_mean = (self.evening_start + self.evening_end) / 2
        evening_std = (self.evening_end - self.evening_start) / 6

        # 随机生成一次通勤
        morning_departure = float(np.clip(np.random.normal(morning_mean, morning_std), self.morning_start, self.morning_end))
        morning_departure = round(morning_departure, 2)
        distance = round(max(0.5, np.random.lognormal(self.commuter_distance_mean, self.commuter_distance_sigma)), 2)
        duration = round(distance / self.avg_speed, 2)
        min_return_time = morning_departure + duration
        evening_return = float(np.random.normal(evening_mean, evening_std))
        evening_return = float(np.maximum(evening_return, min_return_time))
        evening_return = round(float(np.clip(evening_return, 0, 24)), 2)

        start_slot = self.to_slot(morning_departure)
        end_slot = self.to_slot(evening_return)
        if day_index:
            offset = int(day_index) * 96
            start_slot += offset
            end_slot += offset
        return [start_slot, distance, end_slot]

    def generate_short_trips(self, day_index=0):
        """生成短距离出行模式的数据；内部冲突基于当日(0-24h)，返回前再加天偏移。"""
        trips = []
        num_trips = random.randint(2, 3)  # 2-3次出行
        max_attempts = num_trips * 6
        attempts = 0

        while len(trips) < num_trips and attempts < max_attempts:
            attempts += 1
            trip_start = round(random.uniform(self.morning_start, self.evening_end), 2)
            trip_distance = round(self.generate_lognormal_distance(self.short_trip_distance_mean, self.short_trip_distance_sigma), 2)
            trip_duration = self.calculate_trip_duration(trip_distance)
            stay_duration = round(max(0.1, np.random.lognormal(2, 0.5)), 2)
            trip_return = round(self.ensure_time_in_range(trip_start + trip_duration + stay_duration), 2)

            if not self.is_time_conflict(trip_start, trip_return, trips):
                trips.append([self.to_slot(trip_start), trip_distance, self.to_slot(trip_return)])

        trips.sort(key=lambda x: x[0])
        if day_index == 0:
            return trips
        offset = int(day_index) * 96
        trips_with_offset = [[t[0] + offset, t[1], t[2] + offset] for t in trips]
        return trips_with_offset

    def generate_single_day_trips(self, day_index=0):
        """生成单天的出行数据（已废弃：保留以兼容旧接口）。"""
        return [[], []]
    
    def generate_commuter_car_trips_for_days(self, days=14):
        """为一辆通勤车生成多天数据：每天一次通勤行程。"""
        trips = []
        for day_index in range(days):
            trips.append(self.generate_commuter_trip_single(day_index=day_index))
        trips.sort(key=lambda x: x[0])
        return trips

    def generate_short_car_trips_for_days(self, days=14):
        """为一辆短途车生成多天数据：每天2-3次短途行程。"""
        trips = []
        for day_index in range(days):
            trips.extend(self.generate_short_trips(day_index=day_index))
        trips.sort(key=lambda x: x[0])
        return trips

    def generate_all_cars_trips(self, days=14):
        """为所有车辆生成多天数据：前20辆为通勤模式，后30辆为短途模式。返回长度为50的列表，每个元素为对应车辆的出行list。"""
        all_cars = []
        for _ in range(self.commuter_cars):
            all_cars.append(self.generate_commuter_car_trips_for_days(days=days))
        for _ in range(self.short_trip_cars):
            all_cars.append(self.generate_short_car_trips_for_days(days=days))
        return all_cars
    
    def save_data(self, data, filename='envs/ev_trips_data15.json'):
        """保存数据到JSON文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {filename}")
    
    
def main():
    generator = EVTripGenerator()
    
    # 生成14天数据：20辆通勤车 + 30辆短途车，每车为一个list
    print("正在生成电动汽车15天出行数据（20通勤+30短途，按车分组）...")
    array_50d = generator.generate_all_cars_trips(days=20)
    
    # 保存数据
    generator.save_data(array_50d)
    # print(generator.generate_short_trips())

if __name__ == "__main__":
    main()