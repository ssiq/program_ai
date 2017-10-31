#!/usr/bin/env python
#
# Copyright 2007 Neal Norwitz
# Portions Copyright 2007 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenize C++ source code."""

import random
import sys

# Add $ as a valid identifier char since so much code uses it.
# C++0x string preffixes.
# Token types.
from common.code_tokenize import CONSTANT, GetTokens

# Where the token originated from.  This can be used for backtracking.
# It is always set to WHENCE_STREAM in this code.
WHENCE_STREAM, WHENCE_QUEUE = range(2)


def fake_name(name):
    s = list(set(name))
    c = s[random.randint(0, len(s)-1)]
    a = random.randint(0, 1)
    INSERT = 0
    DELETE = 1
    if a == INSERT:
        index = random.randint(0, len(name))
        return name[:index] + c +name[index:]
    elif a == DELETE:
        index = random.randint(0, len(name)-1)
        if index == 0:
            return name[1:]
        elif index == len(name) - 1:
            return name[:-1]
        else:
            return name[:index]+name[index+1:]



def create_fake_cpp_code(source):
    token_list = list(GetTokens(source))
    token_name_list = [t.name for t in token_list]
    # print("token_name_list:{}".format(" ".join(token_name_list)))
    token_name_set = list(set(token_name_list))
    CHANGE = 0
    INSERT = 1
    DELETE = 2
    rid = random.randint(0, 2)
    if rid == CHANGE:
        while True:
            index = random.randint(0, len(token_name_list)-1)
            if token_list[index].token_type == CONSTANT:
                continue
            else:
                ori_token = token_list[index]
                token_name_list[index] = fake_name(token_name_list[index])
                return " ".join(token_name_list), CHANGE, index, ori_token.name
    elif rid == INSERT:
        index = random.randint(0, len(token_name_list))
        insert_token = random.choice(token_name_set)
        token_name_list = token_name_list[:index] + [insert_token] + token_name_list[index:]
        return " ".join(token_name_list), INSERT, index, insert_token
    elif rid == DELETE:
        index = random.randint(0, len(token_name_list)-1)
        delete_token = token_name_list[index]
        if index == 0:
            token_name_list = token_name_list[1:]
        elif index == len(token_name_list)-1:
            token_name_list = token_name_list[:-1]
        else:
            token_name_list = token_name_list[:index]+token_name_list[index+1:]
        return " ".join(token_name_list), DELETE, index, delete_token

if __name__ == '__main__':
    def main(argv):
        """Driver mostly for testing purposes."""
        for filename in argv[1:]:
            with open(filename) as f:
                source = f.read()
            if source is None:
                continue
            for token in GetTokens(source):
                print('%-12s: %s' % (token.token_type, token.name))
                # print('\r%6.2f%%' % (100.0 * index / token.end),)
            sys.stdout.write('\n')
    main(sys.argv)
    with open(sys.argv[1]) as f:
        source = f.read()
    print(create_fake_cpp_code(source))