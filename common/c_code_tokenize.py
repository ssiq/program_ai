from common.pycparser_tokenize.pycparser.pycparser import CParser
from common.pycparser_tokenize.pycparser.pycparser.c_lexer import CLexer


def init_pycparser(lexer=CLexer):
    c_parser = CParser()
    c_parser.build(lexer=lexer)
    return c_parser


def tokenize_by_clex_fn():
    from common.pycparser_tokenize.buffered_clex import BufferedCLex
    c_parser = init_pycparser(lexer=BufferedCLex)
    def tokenize_fn(code):
        tokens = tokenize_by_clex(code, c_parser.clex)
        return tokens
    return tokenize_fn

tokenize_error_count = 0
tokenize_count = 0
def tokenize_by_clex(code, lexer):
    global tokenize_error_count, tokenize_count
    try:
        tokenize_count += 1
        print('tokenize count: ', tokenize_count)
        lexer.reset_lineno()
        lexer.input(code)
        tokens = list(zip(*lexer._tokens_buffer))[0]
        return tokens
    except IndexError as e:
        tokenize_error_count += 1
        # print('token_buffer_len:{}'.format(len(lexer._tokens_buffer)))
        return None
    except Exception as a:
        print(a)
        tokenize_error_count += 1
        return None