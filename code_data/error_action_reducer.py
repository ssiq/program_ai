import random
import more_itertools
from toolz.sandbox.core import unzip
import functools

from common import util
from common.new_tokenizer import keywords, operators
from code_data.constants import pre_defined_cpp_token
from code_data.token_level_fake_code import fake_name


class NotFoundChangePositionException(Exception):
    pass


CHANGE = 0
INSERT = 1
DELETE = 2

def create_identifier_set(tokens, keyword_set=pre_defined_cpp_token):
    tokens_value = [tok.value for tok in tokens]
    tokens_value_set = set(filter(lambda x: not isinstance(x, list), tokens_value))
    identify_set = tokens_value_set - keyword_set
    return identify_set


def action_type_random(type_list=(5, 1, 4)):
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
    return [(type, pos, token_pos, from_char, to_char)]

def _catch_exception_wrapper(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            res = func(*args, **kwargs)
            print("Find the place")
            return res
        except NotFoundChangePositionException:
            print("Not found the place")
            return None

    return wrapper


def position_random_with_filter(tokens, filer_fn, window_size=1):
    windowed_tokens = more_itertools.windowed(tokens, window_size)
    # for t in tokens:
    #     print("tokens:{}, {}".format(t.value, filer_fn([t])))
    random_pos_list = unzip(filter(lambda x:filer_fn(x[1]), enumerate(windowed_tokens)))
    if random_pos_list:
        random_pos_list = list(random_pos_list[0])
    else:
        raise NotFoundChangePositionException
    pos = random.sample(random_pos_list, k=1)[0]
    token = tokens[pos]
    return token.lexpos, pos


@_catch_exception_wrapper
def create_undeclared_identifier(code:str, tokens):
    i_type = CHANGE
    identifier_set = create_identifier_set(tokens, pre_defined_cpp_token)
    filter_fn = lambda x: not isinstance(x[0].value, list) and x[0].value in identifier_set
    return create_error(filter_fn, i_type, tokens, 1)

@_catch_exception_wrapper
def delete_brace(code:str, tokens):
    i_type = DELETE
    braces = {r"{", r"}", r"(", r")", }
    filter_fn = lambda x: not isinstance(x[0].value, list) and x[0].value in braces
    return create_error(filter_fn, i_type, tokens, 1)

@_catch_exception_wrapper
def delete_a_pair_of_braces(code:str, tokens):
    i_type = DELETE
    filter_fn = lambda x: x[0].value == r'(' and x[1].value == r')'
    window_size = 2
    return  create_error(filter_fn, i_type, tokens, window_size=window_size)

@_catch_exception_wrapper
def delete_semicolon(code:str, tokens):
    i_type = DELETE
    filter_fn = lambda x:x[0].value == r';'
    return create_error(filter_fn, i_type, tokens, window_size=1)

@_catch_exception_wrapper
def delete_return_fn(code:str, tokens):
    i_type = DELETE
    begin_fn = lambda x: x[0].value == r'return'
    end_fn = lambda x: x.value == r";"
    return create_error((begin_fn, end_fn), i_type, tokens, window_size=1, filter_fn_is_tuple=True)

@_catch_exception_wrapper
def change_between_pointer_and_reference(code:str, tokens):
    i_type = random.sample([DELETE, INSERT], 1)[0]
    if i_type == DELETE:
        filter_fn = lambda x: x[0].value == r'*' or x[0].value == r'&'
        return create_error(filter_fn, i_type, tokens)
    elif i_type == INSERT:
        identifier_set = create_identifier_set(tokens, pre_defined_cpp_token)
        filter_fn = lambda x: not isinstance(x[0].value, list) and x[0].value in identifier_set
        insert_value = random.sample([r'*', r'&'], 1)[0]
        return create_error(filter_fn, i_type, tokens, change_value=insert_value)



def create_error(filter_fn, i_type, tokens, window_size=1, filter_fn_is_tuple=False, change_value=None):
    identifier_set = create_identifier_set(tokens, pre_defined_cpp_token)
    if not filter_fn_is_tuple:
        pos, token_pos = position_random_with_filter(tokens,
                                                     filter_fn,
                                                     window_size=window_size)
        token_pos_list = list(range(token_pos, token_pos + window_size))
    else:
        begin_fn = filter_fn[0]
        end_fn = filter_fn[1]
        pos, token_pos = position_random_with_filter(tokens,
                                                     begin_fn,
                                                     window_size=window_size)
        token_pos_list = [token_pos]
        for i, token in enumerate(tokens[token_pos+1:]):
            token_pos_list.append(i+token_pos+1)
            if end_fn(token):
                break

    pos_list = [tokens[p].lexpos for p in token_pos_list]
    def c_res(i_pos, i_token_pos):
        from_char, from_char_type = create_from_char(tokens, i_type, i_token_pos)
        if change_value is not None:
            to_char = change_value
        else:
            to_char = to_char_random(i_type, from_char, list(identifier_set))
        return (i_type, i_pos, i_token_pos, from_char, to_char)
    # return [list(res) for res in unzip([c_res(p, t_p) for p, t_p in zip(pos_list, token_pos_list)])]
    return [c_res(p, t_p) for p, t_p in zip(pos_list, token_pos_list)]


def create_error_action_fn():
    i = util.weight_choice(list(zip(*error_creator_list))[2])
    return error_creator_list[i][1]

error_creator_list = [
    ("RANDOM", random_creator, 1),
    ("RANDOM", random_creator, 1),
    # ("Undeclared_identifier", create_undeclared_identifier, 1),
    # ("delete_brace", delete_brace, 1),
    # ("delete_a_pair_of_braces", delete_a_pair_of_braces, 1),
    # ("delete_semicolon", delete_semicolon, 1),
    # ("delete_return_fn", delete_return_fn, 1),
    # ("change_between_pointer_and_reference", change_between_pointer_and_reference, 1)
]