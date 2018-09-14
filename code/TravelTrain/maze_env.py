import sys
import time

import numpy as np

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

action_test = [[1, 2], [1, 4], [2, 2], [2, 4], [4, 1]]

action_reward = {"1,2": 1, "1,4": 2, "2,2": 3, "2,4": 4, "4,1": 5}


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        # 所有的选择状态，迷宫是上下左右四个选项
        self.n_actions = len(action_test)
        # 输入数据的维度
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()
        # 每次训练结束的标志，相当于迷宫出口
        self.endState = ""

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # # hell
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black')
        # # hell
        # # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # # self.hell2 = self.canvas.create_rectangle(
        # #     hell2_center[0] - 15, hell2_center[1] - 15,
        # #     hell2_center[0] + 15, hell2_center[1] + 15,
        # #     fill='black')
        #
        # # create oval
        # oval_center = origin + UNIT * 2
        # self.oval = self.canvas.create_oval(
        #     oval_center[0] - 15, oval_center[1] - 15,
        #     oval_center[0] + 15, oval_center[1] + 15,
        #     fill='yellow')

        for x, y in action_test:
            oval_center = origin + np.array([UNIT * (x - 1), UNIT * (y - 1)])
            self.canvas.create_oval(
                oval_center[0] - 15, oval_center[1] - 15,
                oval_center[0] + 15, oval_center[1] + 15,
                fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        max_reward = -1
        # 将当前最高评价的行为作为结束的标志
        for state, reward in action_reward.items():
            if reward > max_reward:
                max_reward = reward
                self.endState = state
                # return observation
        return np.array([1, 1])

    def step(self, action, observation):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        # if action == 0:  # up
        #     if s[1] > UNIT:
        #         base_action[1] -= UNIT
        # elif action == 1:  # down
        #     if s[1] < (MAZE_H - 1) * UNIT:
        #         base_action[1] += UNIT
        # elif action == 2:  # right
        #     if s[0] < (MAZE_W - 1) * UNIT:
        #         base_action[0] += UNIT
        # elif action == 3:  # left
        #     if s[0] > UNIT:
        #         base_action[0] -= UNIT


        origin = np.array([20, 20])
        dst_center = origin + np.array([UNIT * (action_test[action][0] - 1), UNIT * (action_test[action][1] - 1)])
        # 偏移值
        base_action = np.array([dst_center[0] - (s[0] + 15), dst_center[1] - (s[1] + 15)])
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state
        current_state = '%s,%s' % (observation[0], observation[1])

        if current_state in action_reward.keys():
            reward = action_reward['%s,%s' % (observation[0], observation[1])]
        else:
            reward = 0

        if current_state == self.endState:
            done = True
        else:
            done = False

        # reward function
        # if next_coords == self.canvas.coords(self.oval):
        #     reward = 1
        #     done = True
        # elif next_coords in [self.canvas.coords(self.hell1)]:
        #     reward = -1
        #     done = True
        # else:
        #     reward = 0
        #     done = False

        # 将当前点与目标点的向量作为新的状态
        s_ = np.array(action_test[action])
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()
