import random

from common import util
from common.new_tokenizer import keywords, operators
from code_data.constants import pre_defined_cpp_token
from code_data.token_level_fake_code import fake_name

CHANGE = 0
INSERT = 1
DELETE = 2

def create_identifier_set(tokens, keyword_set=pre_defined_cpp_token):
    tokens_value = [tok.value for tok in tokens]
    tokens_value_set = set(filter(lambda x: not isinstance(x, list), tokens_value))
    identify_set = tokens_value_set - keyword_set
    return identify_set


def action_type_random(type_list=(5, 4, 1)):
    i = util.weight_choice(type_list)
    return i

def position_random(tokens, action_type):
    pos = -1
    while pos < 0 or (action_type != INSERT and isinstance(tokens[pos].value, list)):
        if action_type == INSERT:
            pos = random.randint(0, len(tokens))
        else:
            pos = random.randint(0, len(tokens)-1)
    if pos == len(tokens):
        char_pos = tokens[pos-1].lexpos + len(tokens[pos-1].value)
    else:
        char_pos = tokens[pos].lexpos


    return char_pos, pos

def to_char_random(act_type, from_char, identifier_list, change_type_list=(4, 1), insert_sample_weight=(1, 3, 6)):
    OTHER_WORD = 0
    CONFUSE_WORD = 1
    INSERT_WORD = 2
    to_char = ''
    change_type = util.weight_choice(change_type_list)
    if from_char == '':
        change_type = OTHER_WORD
    if change_type == OTHER_WORD:
        if from_char in identifier_list:
            to_char = random.sample(identifier_list, 1)[0]
        elif from_char in keywords.keys():
            to_char = random.sample(list(keywords.keys()), 1)[0]
        elif from_char in operators.values():
            to_char = random.sample(list(operators.values()), 1)[0]
        else:
            i = util.weight_choice(insert_sample_weight)
            if i == 0:
                range_list = identifier_list
            elif i == 1:
                range_list = list(keywords.keys())
            else:
                range_list = list(operators.values())
            to_char = random.sample(range_list, 1)[0]
    else:
        to_char = fake_name(from_char)
    if act_type == DELETE:
        to_char = ''
    return to_char


def create_from_char(tokens, type, token_pos):
    if type == INSERT:
        from_char = ''
        from_char_type = ''
    else:
        from_char = tokens[token_pos].value
        from_char_type = tokens[token_pos].type
    return from_char, from_char_type

def random_creator(code:str, tokens):
    type = action_type_random()
    pos, token_pos = position_random(tokens, type)
    from_char, from_char_type = create_from_char(tokens, type, token_pos)
    identifier_set = create_identifier_set(tokens, pre_defined_cpp_token)
    to_char = to_char_random(type, from_char, list(identifier_set))
    return (type, pos, token_pos, from_char, to_char)


def create_error_action_fn():
    i = util.weight_choice(list(zip(*error_creator_list))[2])
    return error_creator_list[i][1]

error_creator_list = [
    ("RANDOM", random_creator, 1),
    ("RANDOM", random_creator, 1),
]