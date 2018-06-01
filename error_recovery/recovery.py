from pycparser.pycparser.c_parser import CParser
from common.pycparser_tokenize.buffered_clex import BufferedCLex

import types
import inspect
import re
import functools
import common.deepcopy_with_re as copy
from abc import ABC, abstractmethod


def create_is_parse_fn():
    pattern = re.compile(r"p_.*")

    def is_p_fn(name):
        return pattern.match(name)
    return is_p_fn


class BaseRecoveryFramework(ABC):
    def __init__(self,
                 lex_optimize=True,
                 lexer=BufferedCLex,
                 lextab='pycparser.lextab',
                 yacc_optimize=True,
                 yacctab='pycparser.yacctab',
                 yacc_debug=False,
                 taboutputdir='',
                 ):
        self.parser = CParser()
        is_parse_fn = create_is_parse_fn()
        parse_fn_tuple_list = filter(lambda x: is_parse_fn(x[0]) and x[0] != "p_error", inspect.getmembers(self.parser))
        self.history = []
        self.index = 0

        for k, v in parse_fn_tuple_list:
            # print("{}:{}".format(k, v))
            new_method = types.MethodType(self.patch_p_fn(v), self.parser)
            setattr(self.parser, k, new_method)

        self.parser.parse = types.MethodType(self.patch_parse_fn(self.parser.parse), self.parser)

        self.parser.p_error = types.MethodType(self.patch_p_error_fn(), self.parser)

        self.parser.build(
            lex_optimize,
            lexer,
            lextab,
            yacc_optimize,
            yacctab,
            yacc_debug,
            taboutputdir
        )

        self.parser.clex.add_history_fn = self._create_add_history_fn()

    def __getattr__(self, item):
        return getattr(self.parser, item)

    def _create_add_history_fn(self):
        def add_history():
            self.history.append(copy.deepcopy(self.parser))
            print("{}:{}".format(self.index, self.history[-1].cparser.symstack))
            self.index += 1
        return add_history

    def patch_parse_fn(self, parse):
        @functools.wraps(self.parser.parse)
        def patched_parse(parser_self, *args, **kwargs):
            self.index = 0
            self.history = []
            return parse(*args, **kwargs)

        return patched_parse

    def patch_p_fn(self, fn):
        @functools.wraps(fn)
        def wrapper(parser_self, p):
            self.history.append(copy.deepcopy(parser_self))
            # print("{}:{}, {}".format(self.index, fn.__name__, self.history[-1].cparser.symstack))
            self.index += 1
            return fn(p)

        # assert wrapper.__name__ == fn.__name__
        # assert wrapper.__doc__ == fn.__doc__
        return wrapper

    @abstractmethod
    def _p_error(self, p):
        pass

    def patch_p_error_fn(self):
        @functools.wraps(self.parser.p_error)
        def wrapper(parse_self, p):
            return self._p_error(p)
        return wrapper