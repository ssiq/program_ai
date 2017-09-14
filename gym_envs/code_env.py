import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete, Tuple
import sys
import re
import numpy
import os


class CodeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_STEP = -1

    def __init__(self):
        self.max_step = 200
        self.reset()

    def _step(self, action: list) -> Tuple:
        if not self.last_reward:
            self.last_reward = 0
        if not self.total_rewards:
            self.total_rewards = 0

        self.last_action = action
        self.count_steps += 1

        code = self.code_string

        pos = action[0]
        cha = action[1]
        if pos % 2 == 0:
            pos = pos/2
            if pos == 0:
                code = cha + code
            elif pos == len(code):
                code = code + cha
            else:
                code = code[:pos] + cha +code[pos+1:]
        elif pos % 2 == 1:
            pos = int(pos/2) + 1
            print(pos)
            code = code[:pos-1] + cha + code[pos + 1:]

        reward = 0
        info = {}
        if self.count_steps > self.max_step:
            done = True
            reward = -1
        elif self._compile_code():
            done = True
            reward = 100 - self._diff_between_codes()
        else:
            done = False
            reward = -1
        self.code_string = code
        self.last_reward = reward
        self.total_rewards += reward
        obs = self._get_obs()
        return (obs, reward, done, info)

    def _reset(self) -> str:
        self.original_code = ''
        self.code_string = ''
        self.last_action = None
        self.last_reward = None
        self.total_rewards = None
        self.code_string = self.original_code
        self.count_steps = 0
        obs = self._render_observation()
        return obs

    def _render_observation(self) -> str:
        header = "------------------- Step {} -------------------\n".format(self.count_steps)
        code = "{} \n\n".format(self.code_string)
        last_re = "Last Reward : {} \n".format(self.last_reward)
        total_re = "Total Reward : {} \n".format(self.total_rewards)
        return header+code+last_re+total_re+'\n'

    def _get_obs(self) -> str:
        return self.code_string

    def _render(self, mode: str='human', close: bool=False):
        outfile = sys.stdout
        outfile.write(self._render_observation())
        return outfile

    def render_file(self, mode: str='human', close: bool=False, outfile=sys.stdout):
        if not outfile:
            outfile = sys.stdout
        outfile.write(self._render_observation())
        return outfile

    def _get_next_code(self):
        return ""

    def _preprocess_code(self, code):
        pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
        mat = re.findall(pattern, code)
        processed_code = ' '.join(mat)
        return processed_code

    def _diff_between_codes(self):
        lcs = self.find_lcseque(self.original_code, self.code_string)
        return (len(lcs)*100)/len(self.original_code)

    def _compile_code(self) -> bool:
        return False

    def find_lcseque(self, s1, s2):
        # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
        m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
        # d用来记录转移方向
        d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

        for p1 in range(len(s1)):
            for p2 in range(len(s2)):
                if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                    m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                    d[p1 + 1][p2 + 1] = 'ok'
                elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                    m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                    d[p1 + 1][p2 + 1] = 'left'
                else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                    m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                    d[p1 + 1][p2 + 1] = 'up'
        (p1, p2) = (len(s1), len(s2))

        s = []
        while m[p1][p2]:  # 不为None时
            c = d[p1][p2]
            if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
                s.append(s1[p1 - 1])
                p1 -= 1
                p2 -= 1
            if c == 'left':  # 根据标记，向左找下一个
                p2 -= 1
            if c == 'up':  # 根据标记，向上找下一个
                p1 -= 1
        s.reverse()
        return ''.join(s)
