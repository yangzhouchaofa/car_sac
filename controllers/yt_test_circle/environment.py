import random
import math
import os
import pickle
import transforms3d as tf3
from scipy.spatial.transform import Rotation as R
from main import train
from controller import Robot
import numpy as np
from gym.spaces import Box, Discrete
from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from utilities import normalize_to_range, normalize
from main import STEPS_PER_EPISODE


class Hunter(RobotSupervisor):

    def __init__(self):
        super().__init__(timestep=1000)
        #       speed   x    y    angle   dx     dy    angle_d   8
        # min   -1.4  -6.2 -4.92  -3.14  -6.2  -4.92   -3.14
        # max    1.4  1.82  1.99   3.14  1.82   1.99    3.14
        self.observation_space = 14
        # speed        turn
        # (-1,1)*1.4  (-1,1)*0.4
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float64)
        self.DistanceRewardInterval = 0.15
        self.DistanceThreshold = 0.10
        self.RotationThreshold = 0.20
        self.CurrentSpeed = 0
        self.w = 0
        self.w_list = []
        self.SpeedTheshold = 0.1
        self.park_count = 0
        self.crash_count = 0
        self.Speed_list = []
        self.avg_speed = 0
        self.Repeat_reward = []
        self.complete = 0

        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = "yt"  # the name of this env.
        self.max_step = STEPS_PER_EPISODE  # the max step of each episode
        self.state_dim = self.observation_space  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = -200  # episode return is between (-1600, 0)

        print(self.timestep)
        # self.timestep = int(self.getBasicTimeStep())
        self.gps_back = self.getDevice('gpsBack')
        self.gps_front = self.getDevice('gpsFront')
        self.laser = self.getDevice('rslidar')
        # self.ds = self.getDevice('ds')
        self.laserRange = []
        self.position = []
        self.positionOld = []
        self.disReward = float("inf")
        self.disRewardOld = float("inf")
        self.angReward = float("inf")

        self.front_ds = 0.6  # front safe distance of car
        self.side_ds = 0.4  # beside safe distance of car
        self.behide_ds = 0.6  # back safe distance of car
        self.rec_degree = (round(math.atan(self.side_ds / self.behide_ds) * 180 / math.pi),
                           180 - round(math.atan(self.side_ds / self.behide_ds) * 180 / math.pi) - round(
                               math.atan(self.side_ds / self.front_ds) * 180 / math.pi),
                           round(math.atan(
                               self.side_ds / self.front_ds) * 180 / math.pi))  # use rectangle shape around the car to avoid obstacle

        self.steps = 0
        self.steps_threshold = 6000

        # self.tarPosition = [-0.966, -1.52, -1.57]  # x, z, angle(-3.14,3.14)
        self.tarPosition = [2, 0.5, 3.14]  # x, z, angle(-3.14,3.14)
        self.wheel_front = []
        for wheelName in ['wheel_lm_motor', 'wheel_rm_motor']:
            wheel = self.getDevice(wheelName)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.wheel_front.append(wheel)

        self.wheel_rear = []
        for wheelName in ['wheel_lm_motor', 'wheel_rm_motor']:
            wheel = self.getDevice(wheelName)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0)
            self.wheel_rear.append(wheel)

        # self.ds.enable(self.timestep)
        self.laser.enable(self.timestep)  # Frequency:10hz
        self.laser.enablePointCloud()
        self.gps_back.enable(self.timestep)
        self.gps_front.enable(self.timestep)

        self.stepsPerEpisode = 10000  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved

    def motorIn(self, speed, steer):  # left+ right-
        # for wheelFront in self.wheel_front:
        #     wheelFront.setPosition(steer)
        for wheelRear in self.wheel_rear:
            wheelRear.setPosition(float('inf'))  # must add this
        self.wheel_rear[0].setVelocity(speed)
        self.wheel_rear[1].setVelocity(steer)

    def SpeedChange(self):
        self.Speed_list.append(self.CurrentSpeed)
        if (len(self.Speed_list) <= 1) or (len(self.w_list) <= 1):
        # if (len(self.Speed_list) <= 1):
            return False
        elif (self.Speed_list[-2] * self.Speed_list[-1] < -0.1) or \
             (abs(self.Speed_list[-2] + self.Speed_list[-1]) < 0.2 and self.w_list[-2] * self.w_list[-1] < -0.1):
        # elif (self.Speed_list[-2] * self.Speed_list[-1] < -0.1):
            return True
        else:
            return False

    def get_observations(self):
        # print('observation')

        observation = []

        self.laser.enable(self.timestep)
        self.laserRange = self.laser.getLayerRangeImage(0)

        # observation.append(normalize_to_range(min(self.laserRange), 0.0, 8.0, 0.0, 1.0))
        # print(min(self.laserRange))

        self.position = []
        gpsBack = self.gps_back.getValues()
        gpsFront = self.gps_front.getValues()
        if train == 'train':
            noise_laser = np.random.normal(0, 0.01)
            noise_odom = np.random.normal(0, 0.03)
        elif train == 'test':
            noise_laser = 0
            noise_odom = 0

        self.position.append((gpsBack[0] + gpsFront[0]) / 2)  # x
        self.position.append((gpsBack[2] + gpsFront[2]) / 2)  # y

        self.position.append(-math.atan2(gpsFront[2] - gpsBack[2], gpsFront[0] - gpsBack[0]))

        dx = self.position[0] - self.tarPosition[0]
        dy = self.position[1] - self.tarPosition[1]
        self.disReward = math.sqrt(dx * dx + dy * dy)

        # angReward varies from [-pi,pi] relative to tarPosition and left positive / right negtive
        if self.tarPosition[2] >= 0:
            if self.position[2] - self.tarPosition[2] <= -math.pi:
                self.angReward = self.position[2] - self.tarPosition[2] + 2 * math.pi
            else:
                self.angReward = self.position[2] - self.tarPosition[2]
        else:
            if self.position[2] - self.tarPosition[2] >= math.pi:
                self.angReward = self.position[2] - self.tarPosition[2] - 2 * math.pi
            else:
                self.angReward = self.position[2] - self.tarPosition[2]
        #       speed   x    y    angle   dx     dy    angle_d   8
        # min   -1.4  -6.2 -4.92  -3.14  -6.2  -4.92   -3.14
        # max    1.4  1.82  1.99   3.14  1.82   1.99    3.14
        # 当前位置：
        observation.append(normalize_to_range(np.clip(self.position[0] + noise_odom, 0.36, 4.25), 0.36, 4.25, -1, 1))
        observation.append(normalize_to_range(np.clip(self.position[1] + noise_odom, -0.22, 2.61), -0.22, 2.61, -1, 1))
        observation.append(normalize_to_range(np.clip(self.position[2], -3.14, 3.14), -3.14, 3.14, -1, 1))
        observation.append(normalize_to_range(np.clip(self.CurrentSpeed, -1, 1), -1, 1, -1, 1))
        observation.append(normalize_to_range(np.clip(self.w, -1, 1), -1, 1, -1, 1))
        observation.append(normalize_to_range(np.clip(self.disReward, -4.8, 4.8), -4.8, 4.8, -1, 1))
        observation.append(normalize_to_range(np.clip(self.angReward, -3.14, 3.14), -3.14, 3.14, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[0], 0.36, 4.25), 0.36, 4.25, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[1], -0.22, 2.61), -0.22, 2.61, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[2], -3.14, 3.14), -3.14, 3.14, -1, 1))
        # observation.append(min(self.laserRange))

        # ******************************************************************************
        self.laser_choose = [0, 10, 20, 30, 40, 50, 60, 70, 80,
                             90, 100, 110, 120, 130, 140, 150, 160, 170,
                             180, 190, 200, 210, 220, 230, 240, 250, 260,
                             270, 280, 290, 300, 310, 320, 330, 340, 350]
        self.crash_range = [0.353, 0.358, 0.375, 0.408, 0.456, 0.376, 0.332, 0.306, 0.292,
                            0.288, 0.292, 0.306, 0.332, 0.376, 0.456, 0.408, 0.375, 0.358,
                            0.353, 0.358, 0.375, 0.408, 0.456, 0.376, 0.332, 0.306, 0.292,
                            0.288, 0.292, 0.306, 0.332, 0.376, 0.456, 0.408, 0.375, 0.358]
        for choose in range(len(self.laser_choose)):
            self.laserRange[self.laser_choose[choose]] -= self.crash_range[choose]
        for choose in self.laser_choose:
            observation.append(normalize_to_range(np.clip(self.laserRange[choose] + noise_laser, 0, 4), 0, 4, 0, 1))
        # ******************************************************************************

        return observation

    def apply_action(self, action):  # speed   turn
        for i in range(2):
            action[i] = np.clip(action[i], -1, 1)
        self.motorIn(action[0] * 5, action[1] * 5)
        # print(action)
        self.CurrentSpeed = (action[0] + action[1]) / 2
        self.w = (action[0] - action[1]) / 2

    def get_reward(self, action):
        reward = 0
        deltaDis = self.disRewardOld - self.disReward
        angleDis = abs(self.angReward)
        flag = 0

        reward -= 0.1
        if deltaDis > 0 and self.car_crash() is False:
            if (int)(self.disReward / self.DistanceRewardInterval) < (int)(
                    self.disRewardOld / self.DistanceRewardInterval):
                reward += 0.1 * normalize_to_range(np.clip(2.25 - self.disReward, 0, 2.25), 0, 2.25, 0, 1)
            if angleDis < math.pi / 2:
                reward += 0.1 * normalize_to_range(math.pi / 2 - angleDis, 0, math.pi / 2, 0, 1)

        else:
            # note that '/ 3' is a hard coded value here, which I introduced after tuning the penalty to occur less frequently than
            # the reward, in order to not 'scare' the AI of performing corrective maneuvers where it has to first increase the
            # distance to the target parking spot.
            if (int)(self.disReward / self.DistanceRewardInterval / 2) > (int)(
                    self.disRewardOld / self.DistanceRewardInterval / 2):
                reward += -1

        # Check task completion (= position and rotation lower than threshold)
        if self.car_crash() is True:
            reward += -5

        if self.disReward <= self.DistanceThreshold and self.car_crash() is False:
            temp = 0
            if abs(normalize_to_range(np.clip(self.CurrentSpeed, -1, 1), -1, 1, -1, 1)) <= self.SpeedTheshold \
                    and angleDis < self.RotationThreshold\
                    and (normalize_to_range(np.clip(self.w, -1, 1), -1, 1, -1, 1)) <= self.SpeedTheshold :
                temp += 40
                if angleDis > 0:
                    temp += -40 * normalize(angleDis, 0, self.RotationThreshold)
            reward += temp
        # must add this or the gradient is 'inf', then the loss is 'nan'
        if reward == float('-inf') or reward == float('inf'):
            reward = 0

        if self.SpeedChange() is True:
            reward -= 1
        # if self.CurrentSpeed == 0 and self.disReward >= self.DistanceThreshold:
        #     reward -= 0.1

        self.disRewardOld = self.disReward
        # self.dyOld = self.dy
        return reward

    def car_crash(self):
        for i in range(len(self.laser_choose)):
            if self.laserRange[self.laser_choose[i]] < 0.02:
                # print('laseer_range:',self.laserRange[self.laser_choose[i]],'laser_choose:',self.laser_choose[i])
                return True
        return False

    def is_done(self):
        if self.car_crash() is True:
            self.complete = 0
            return False
        elif self.disReward <= self.DistanceThreshold and \
                abs(normalize_to_range(np.clip(self.CurrentSpeed, -1, 1), -1, 1, -1, 1)) <= self.SpeedTheshold and \
                abs(self.angReward) < self.RotationThreshold and \
                abs(normalize_to_range(np.clip(self.w, -1, 1), -1, 1, -1, 1)) <= self.SpeedTheshold :  # math.pi / 2
            self.complete = 1
            return True
        else:
            self.complete = 0
            return False

    def solved(self):
        # print('solved')
        if len(self.episodeScoreList) > 500:  # Over 500 trials thus far
            if np.mean(self.episodeScoreList[-500:]) > 120.0:  # Last 500 episode scores average value
                return True
        return False

    def get_default_observation(self):
        Obs = [0.0 for _ in range(self.observation_space)]
        return Obs

    def render(self, mode='human'):
        print("render() is not used")

    def get_info(self):
        pass

    def random_initialization(self, rb_node=None, Radius=None, tar=None, next=None):
        self.tarPosition[0] = random.uniform(0.67,4)
        self.tarPosition[1] = random.uniform(0,2.3)
        self.tarPosition[2] = random.uniform(-3.14,3.14)

        # x_range1 = [0.67, 1.5]
        # x_range2 = [1.5, 2.5]
        # x_range3 = [2.5, 4]
        # z_range1 = [0, 2.3]
        # z_range2 = [1.3, 2.3]

        if rb_node == None:
            rb_node = self.getFromDef("yt")

        else:
            rb_node = rb_node
        self.reset()
        self.motorIn(0, 0)

        x = random.uniform(max(self.tarPosition[0] - next, 0.67), min(self.tarPosition[0] + next, 4))
        z = random.uniform(max(self.tarPosition[1] - next, 0), min(self.tarPosition[1] + next, 2.3))

        y = 0.0912155
        self.rand_rotation_y = R.from_euler('z', random.uniform(0, 360),
                                            degrees=True)  # euler to mat return A matrix,which only use random z axes.

        x_ob = -1
        y_ob = -1
        for i in range(5):
            while ((x_ob == -1) or (d2 <= 1) or (d1 <= 1)):
                x_ob = random.uniform(0.2, 4.4)
                y_ob = random.uniform(-0.3, 2.7)
                d1 = (x_ob - x) * (x_ob - x) + (y_ob - y) * (y_ob - y)
                d2 = (x_ob - self.tarPosition[0]) * (x_ob - self.tarPosition[0]) + (y_ob - self.tarPosition[1]) * (
                            y_ob - self.tarPosition[1])

            ob = self.getFromDef("ob%d" % i)
            ob_position = ob.getField("translation")
            ob_position.setSFVec3f([x_ob, 0.0912155, y_ob])
            x_ob = -1
            y_ob = -1

        ob = self.getFromDef("tar")
        ob_position = ob.getField("translation")
        ob_position.setSFVec3f([self.tarPosition[0], 0.0, self.tarPosition[1]])

        INITIAL = [x, y, z]
        trans_field = rb_node.getField("translation")
        trans_field.setSFVec3f(INITIAL)

        rotation_field = rb_node.getField("rotation")
        quaternion = [0.706522, -0.707691, 0, 0]
        trans_mat = tf3.quaternions.quat2mat(quaternion)

        trans_mat = self.rand_rotation_y.apply(trans_mat)
        quaternion = tf3.quaternions.mat2quat(trans_mat)
        # quaternion to axis angle
        angle = 2 * math.acos(quaternion[0])
        x = quaternion[1] / math.sqrt(1 - quaternion[0] * quaternion[0])
        y = quaternion[2] / math.sqrt(1 - quaternion[0] * quaternion[0])
        z = quaternion[3] / math.sqrt(1 - quaternion[0] * quaternion[0])
        axis_angle = [x, y, z, angle]
        rotation_field.setSFRotation(axis_angle)
        Robot.step(self, self.timestep)  # 更新环境
        return self.get_observations()

    def initialization(self, rb_node=None, x=1, z=0, rotation=200, Radius=None, tar=None, next=None):
        if rb_node == None:
            rb_node = self.getFromDef("yt")

        else:
            rb_node = rb_node
        self.reset()
        self.motorIn(0, 0)
        if len(self.obs) != 0:
            for i in range(len(self.obs[0])):
                ob = self.getFromDef("ob%d" % i)
                ob_position = ob.getField("translation")
                ob_position.setSFVec3f([self.obs[0][i][0], 0.0912155, self.obs[0][i][1]])

        ob = self.getFromDef("tar")
        ob_position = ob.getField("translation")
        ob_position.setSFVec3f([self.tarPosition[0], 0.0, self.tarPosition[1]])

        y = 0.2
        self.rand_rotation_y = R.from_euler('z', rotation,
                                            degrees=True)  # euler to mat return A matrix,which only use random z axes.
        INITIAL = [x, y, z]
        trans_field = rb_node.getField("translation")
        trans_field.setSFVec3f(INITIAL)  #

        rotation_field = rb_node.getField("rotation")
        quaternion = [0.707, -0.707, 0, 0]
        trans_mat = tf3.quaternions.quat2mat(quaternion)

        trans_mat = self.rand_rotation_y.apply(trans_mat)
        quaternion = tf3.quaternions.mat2quat(trans_mat)
        # quaternion to axis angle
        angle = 2 * math.acos(quaternion[0])
        x = quaternion[1] / math.sqrt(1 - quaternion[0] * quaternion[0])
        y = quaternion[2] / math.sqrt(1 - quaternion[0] * quaternion[0])
        z = quaternion[3] / math.sqrt(1 - quaternion[0] * quaternion[0])
        axis_angle = [x, y, z, angle]
        rotation_field.setSFRotation(axis_angle)
        Robot.step(self, self.timestep)  # 更新环境
        return self.get_observations()

    def test(self, num):
        self.PATH = '/home/ylc/webots_ros/src/webots_ros/SAC1/data%d'
        self.PICKLE0 = 'laser.pkl'
        self.PICKLE1 = 'gpsfront.pkl'
        self.PICKLE2 = 'gpsback.pkl'
        self.PICKLE3 = 'motor.pkl'
        self.PICKLE4 = 'obs.pkl'
        self.PICKLE = [self.PICKLE0, self.PICKLE1, self.PICKLE2, self.PICKLE3]
        self.obs = []
        self.data = []
        self.num = num

        # 记录障碍物的位置
        with open(os.path.join(self.PATH % self.num, self.PICKLE4), 'rb') as c:
            self.obs.append(pickle.load(c))
        # 记录雷达、gps的值
        for i in range(len(self.PICKLE)):
            with open(os.path.join(self.PATH % self.num, self.PICKLE[i]), 'rb') as c:
                self.data.append(pickle.load(c))

        start = max(self.data[0][0][-1], self.data[1][0][-1], self.data[2][0][-1])

        for i in range(len(self.PICKLE)):
            for j in range(
                    max(len(self.data[0]), len(self.data[1]), len(self.data[2])) - min(len(self.data[0]),
                                                                                       len(self.data[1]),
                                                                                       len(self.data[2]))):
                if (self.data[i][0][-1] < start) and ((start - self.data[i][0][-1]) > 0.05):
                    del (self.data[i][j])

        for i in range(3):
            self.data[i] = self.data[i][::10]

        # 对齐motor数据的时间戳
        data_motor = [0] * len(self.data[1])  # 创建一个数组用来保存查找后的速度
        for i in range(len(self.data[1])):  # 对于查找后的长度
            actual_time = self.data[1][i][-1]
            diff_0 = abs(self.data[3][0][-1] - actual_time)
            data_motor[i] = self.data[3][0]
            for j in range(len(self.data[3])):
                diff_j = abs(self.data[3][j][-1] - actual_time)
                if diff_j < diff_0:
                    data_motor[i] = self.data[3][j]
                    diff_0 = diff_j
        self.data[3] = data_motor

        # 去掉时间戳
        for i in range(len(self.PICKLE)):
            for j in range(len(self.data[i])):
                self.data[i][j] = self.data[i][j][:-1]

        for i in range(5):
            self.data[3].append([0.0, 0.0])
            self.data[2].append(self.data[2][-1])
            self.data[1].append(self.data[1][-1])
            self.data[0].append(self.data[0][-1])

        x = (self.data[1][0][0] + self.data[2][0][0]) / 2
        y = (self.data[1][0][2] + self.data[2][0][2]) / 2
        rotation = (math.atan2(self.data[2][0][2] - self.data[1][0][2],
                               self.data[2][0][0] - self.data[1][0][0])) * 180 / math.pi

        self.initialization(x=x, z=y, rotation=rotation, tar=self.tarPosition, )
        self.test_get_observations(0)
        self.disRewardOld = self.disReward
        for i in range(len(self.data[0])):
            self.test_get_observations(i)
            action = self.action
            self.step(action)

        gpsBack = self.gps_back.getValues()
        gpsFront = self.gps_front.getValues()
        self.tarPosition = []
        self.tarPosition.append((gpsBack[0] + gpsFront[0]) / 2)  # x
        self.tarPosition.append((gpsBack[2] + gpsFront[2]) / 2)  # y
        self.tarPosition.append(-math.atan2(gpsFront[2] - gpsBack[2], gpsFront[0] - gpsBack[0]))


    def test_get_observations(self, i):

        observation = []

        noise_laser = np.random.normal(0, 0.01)
        noise_odom = np.random.normal(0, 0.03)

        position = []
        position.append(self.data[2][i][0])  # x
        position.append(self.data[2][i][2])  # y
        position.append(
            -math.atan2(self.data[1][i][2] - self.data[2][i][2], self.data[1][i][0] - self.data[2][i][0]))
        dx = position[0] - self.tarPosition[0]
        dy = position[1] - self.tarPosition[1]
        self.disReward = math.sqrt(dx * dx + dy * dy)
        self.laserRange = self.data[0][i][0]

        # angReward varies from [-pi,pi] relative to tarPosition and left positive / right negtive
        if self.tarPosition[2] >= 0:
            if position[2] - self.tarPosition[2] <= -math.pi:
                self.angReward = position[2] - self.tarPosition[2] + 2 * math.pi
            else:
                self.angReward = position[2] - self.tarPosition[2]
        else:
            if position[2] - self.tarPosition[2] >= math.pi:
                self.angReward = position[2] - self.tarPosition[2] - 2 * math.pi
            else:
                self.angReward = position[2] - self.tarPosition[2]
        #       speed   x    y    angle   dx     dy    angle_d   8
        # min   -1.4  -6.2 -4.92  -3.14  -6.2  -4.92   -3.14
        # max    1.4  1.82  1.99   3.14  1.82   1.99    3.14
        v = self.data[3][i][0]
        w = self.data[3][i][1]
        # print(v,w,-((v + w * 0.199) / (0.1)), -((v - w * 0.199) / (0.1)))
        self.action = [-((v + w * 0.199) / (0.1)) / 5, -((v - w * 0.199) / (0.1)) / 5]
        self.CurrentSpeed = (self.action[0] + self.action[1]) / 2

        observation.append(normalize_to_range(np.clip(position[0] + noise_odom, 0.36, 4.25), 0.36, 4.25, -1, 1))
        observation.append(normalize_to_range(np.clip(position[1] + noise_odom, -0.22, 2.61), -0.22, 2.61, -1, 1))
        observation.append(normalize_to_range(np.clip(position[2], -3.14, 3.14), -3.14, 3.14, -1, 1))
        observation.append(normalize_to_range(np.clip(self.CurrentSpeed, -1, 1), -1, 1, -1, 1))
        observation.append(normalize_to_range(np.clip(self.w, -1, 1), -1, 1, -1, 1))
        observation.append(normalize_to_range(np.clip(self.disReward, -4.8, 4.8), -4.8, 4.8, -1, 1))
        observation.append(normalize_to_range(np.clip(self.angReward, -3.14, 3.14), -3.14, 3.14, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[0], 0.36, 4.25), 0.36, 4.25, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[1], -0.22, 2.61), -0.22, 2.61, -1, 1))
        observation.append(normalize_to_range(np.clip(self.tarPosition[2], -3.14, 3.14), -3.14, 3.14, -1, 1))
        # observation.append(min(self.laserRange))

        # ******************************************************************************
        self.laser_choose = [0, 10, 20, 30, 40, 50, 60, 70, 80,
                             90, 100, 110, 120, 130, 140, 150, 160, 170,
                             180, 190, 200, 210, 220, 230, 240, 250, 260,
                             270, 280, 290, 300, 310, 320, 330, 340, 350]
        self.crash_range = [0.353, 0.358, 0.375, 0.408, 0.456, 0.376, 0.332, 0.306, 0.292,
                            0.288, 0.292, 0.306, 0.332, 0.376, 0.456, 0.408, 0.375, 0.358,
                            0.353, 0.358, 0.375, 0.408, 0.456, 0.376, 0.332, 0.306, 0.292,
                            0.288, 0.292, 0.306, 0.332, 0.376, 0.456, 0.408, 0.375, 0.358]
        for choose in range(len(self.laser_choose)):
            self.laserRange[self.laser_choose[choose]] -= self.crash_range[choose]
        for choose in self.laser_choose:
            observation.append(normalize_to_range(np.clip(self.laserRange[choose] + noise_laser, 0, 4), 0, 4, 0, 1))
        # ******************************************************************************

        return observation
