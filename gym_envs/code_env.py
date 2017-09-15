import gym
import sys
import re
import numpy as np
import os
from code_data.constants import cpp_tmp_dir, cpp_tmp_filename, cpp_tmp_path, sign_char_dict, char_sign_dict
from code_data.read_data import read_cpp_code_list, read_less_cpp_code_list


class CodeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_STEP = -1

    code_df = None
    code_df_size = 0

    def __init__(self):
        self.max_step = 200
        self.reset()

    def _step(self, action: list) -> tuple:
        if not self.last_reward:
            self.last_reward = 0
        if not self.total_rewards:
            self.total_rewards = 0

        self.last_action = action
        self.count_steps += 1

        code = self.code_string

        pos = action[0]
        cha = sign_char_dict[action[1]]
        if cha == 'plh':
            cha = ''

        if pos % 2 == 0:
            pos = int(pos/2)
            code = code[:pos] + cha + code[pos:]
        elif pos % 2 == 1:
            pos = int(pos/2)
            code = code[:pos] + cha + code[pos + 1:]
        self.code_string = code

        reward = 0
        info = {}
        if self.count_steps > self.max_step:
            done = True
            reward = -1
        elif self._compile_code():
            done = True
            reward = self._diff_between_codes()
        else:
            done = False
            reward = -1
        self.last_reward = reward
        self.total_rewards += reward
        obs = self._get_sign_obs()
        return (obs, reward, done, info)

    def _reset(self) -> str:

        if self.code_df == None:
            self.code_df = read_less_cpp_code_list()
            self.code_df_size = self.code_df.shape[0]

        self.count_steps = 0
        self.original_code = self._preprocess_code(self._get_next_code())
        self.code_string = self.original_code
        self.last_action = None
        self.last_reward = None
        self.total_rewards = None
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

    def _get_sign_obs(self):
        char_list = list(self.code_string)
        sign_list = [ char_sign_dict[x] for x in char_list]
        return sign_list

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
        ind = np.random.randint(0, self.code_df_size)
        code = self.code_df['code'].iloc[ind]
        return code

    def _preprocess_code(self, code):
        pattern = re.compile('''('.*?'|".*?"|[^ \t\r\f\v"']+)''')
        mat = re.findall(pattern, code)
        processed_code = ' '.join(mat)
        return processed_code

    def _diff_between_codes(self):
        lcs = self.find_lcseque(self.original_code, self.code_string)
        if len(self.original_code) == 0:
            return 0
        return int((len(lcs)*100)/len(self.original_code))

    def _compile_code(self) -> bool:
        if not os.path.exists(cpp_tmp_dir):
            os.makedirs(cpp_tmp_dir)
        f = open(cpp_tmp_path, 'w')
        f.write(self.code_string)
        f.close()
        res = os.system('g++ -O0 -fsyntax-only {} >/dev/null 2>&1'.format(cpp_tmp_path))
        if res == 0:
            return True
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
