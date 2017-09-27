import gym
import sys
import re
import numpy as np
import os
from code_data.constants import cpp_tmp_dir, cpp_tmp_filename, cpp_tmp_path, sign_char_dict, char_sign_dict
from code_data.read_data import read_cpp_code_list, read_less_cpp_code_list, read_length_cpp_code_list
import math
import time
from database.database_util import insertEpisodes, insertStepInfoMany, backup


class CodeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_STEP = -1

    def __init__(self, max_step=200, max_code_length=100):
        self.max_step = max_step
        self.max_code_length = max_code_length
        self.esp_count = 0
        self.resolved_count = 0

        self.code_df = None
        self.code_df_size = 0

        self.backup_db()
        self.reset()

    def set_max_code_length(self, max_code_length):
        self.max_code_length = max_code_length

    def backup_db(self):
        backup()

    def _step(self, action: int) -> tuple:
        if not self.last_reward:
            self.last_reward = 0
        if not self.total_rewards:
            self.total_rewards = 0

        self.last_action = action
        self.count_steps += 1

        code = self.code_string

        pos_a = int(action/len(char_sign_dict))
        pos = pos_a
        cha = sign_char_dict[action%len(char_sign_dict)]
        if cha == 'plh':
            cha = ''

        if pos >= (2 * len(code)):
            code = code + cha
        elif pos % 2 == 0:
            pos = int(pos/2)
            code = code[:pos] + cha + code[pos:]
        elif pos % 2 == 1:
            pos = int(pos/2)
            code = code[:pos] + cha + code[pos + 1:]
        self.code_string = code

        reward = 0
        info = {}
        resolved = 0
        if self.count_steps > self.max_step:
            done = True
            reward = -1
        elif self._compile_code():
            done = True
            resolved = 1
            reward = self._diff_between_codes()
        else:
            done = False
            reward = -1
        self.last_reward = reward
        self.total_rewards += reward
        obs = self._get_sign_obs()
        donenum = 1 if done else 0
        step_info_item = self.produce_step_info(pos_a, cha, reward, donenum)
        self.step_memory.append(step_info_item)
        if done:
            self.store_esp(resolved)
            self.deal_resolved(resolved)
        return (obs, reward, done, info)

    def _reset(self) -> list:
        self.esp_count += 1
        if self.code_df is None:
            # self.code_df = read_length_cpp_code_list(self.max_code_length)
            self.code_df = read_cpp_code_list()
            self.code_df_size = self.code_df.shape[0]

        self.count_steps = 0
        codeid, code = self._get_next_code()
        self.original_code = self._preprocess_code(code)
        self.code_string = self.original_code
        self.last_action = None
        self.last_reward = None
        self.total_rewards = None
        obs = self._get_sign_obs()

        self.episode = {"episodeid": self.esp_count, "starttime": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        "endtime": "", "totalstep": 0, "totalreward": 0, "resolved": 0, "codeid": codeid,
                        "originalcode": self.original_code, "endcode": ""}
        self.step_memory = []

        return obs

    def deal_resolved(self, resolved):
        if resolved == 1:
            self.resolved_count += 1
            if self.resolved_count > 20:
                self.max_code_length += 10
        else:
            self.resolved_count = 0

    def produce_step_info(self, actionpos, actioncha, rew, don):
        step_info = []
        step_info.append(self.esp_count)
        step_info.append(self.count_steps)
        step_info.append(actionpos)
        step_info.append(actioncha)
        step_info.append(rew)
        step_info.append(don)
        return step_info

    def store_esp(self, compile):
        self.episode["endtime"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.episode["totalstep"] = self.count_steps
        self.episode["totalreward"] = self.total_rewards
        self.episode["resolved"] = compile
        self.episode["endcode"] = self.code_string

        insertEpisodes(None, str(self.episode["episodeid"]), self.episode["starttime"], self.episode["endtime"],
                       str(self.episode["totalstep"]), str(self.episode["totalreward"]), str(self.episode["resolved"]),
                       str(self.episode["codeid"]), self.episode["originalcode"], self.episode["endcode"])
        insertStepInfoMany(self.step_memory)

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
        sign_list = [char_sign_dict[x] for x in char_list]
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
        code_len = math.inf
        code = ''
        id = ''
        while code_len > self.max_code_length:
            ind = np.random.randint(0, self.code_df_size)
            id = self.code_df['id'].iloc[ind]
            code = self.code_df['code'].iloc[ind]
            code_len = len(code)
            if self.check_vaild(code) == 0:
                code_len = math.inf
        return id, code

    def check_vaild(self, code):
        for c in code:
            if c not in char_sign_dict:
                return 0
        return 1

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
