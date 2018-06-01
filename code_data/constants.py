import os
from common.new_tokenizer import keywords, operators

ROOT_PATH = '/home/lf/Project/program_ai'
from common.util import reverse_dict

# ROOT_PATH = r'G:\Project\program_ai'
# ROOT_PATH = r"D:\Machine Learning\program_ai"
DATABASE_PATH = os.path.join(ROOT_PATH, 'data/train.db')
BACKUP_PATH = os.path.join(ROOT_PATH, 'backup/')
LOG_PATH = os.path.join(ROOT_PATH, 'logs/')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'checkpoint/')
SQL_DIR_PATH = os.path.join(ROOT_PATH, 'database/sql/')
EPISODES = 'episodes'
STEP_INFO = 'step_info'
FAKE_ERROR_CODE = 'fake_error_code'
FAKE_ERROR_TOKEN_CODE = 'fake_error_token_code'
FAKE_CODE_RECORDS = 'fake_code_records'
RANDOM_TOKEN_CODE_RECORDS = 'random_token_code_records'
COMMON_ERROR_TOKEN_CODE_RECORDS = 'common_error_token_code_records'

STUDENT_BUILD_INFO = 'student_build_info'
BUILD_ERROR_STAT = 'build_error_stat'
STUDENT_TEST_BUILD_ERROR_STAT = 'student_test_build_error_stat'
TEST_CODE_RECORDS = 'test_code_records'
TEST_EXPERIMENT_RECORDS = 'test_experiment_records'
DEEPFIX_TABLE = 'Code'

LOG_DIR = os.path.join('.', 'log')
DEBUG_LOG_PATH = os.path.join(LOG_DIR, 'debug')
OUTPUT_LOG_PATH = os.path.join(LOG_DIR, 'output')
DOT_PATH = os.path.join(LOG_DIR, 'dot')
OBJECTLENGTH_PATH = os.path.join(LOG_DIR, 'objectlength')

debug_logger_name_list = ['record', 'growth', 'objectlength']
output_logger_name_list = []

spider_key_path = 'key/id_rsa_spider'
remote_db_path = '/home/lf/DataCrawl/scrapy_OJ/release_data/scrapyOJ.db'
local_db_path = os.path.join(ROOT_PATH, 'data/scrapyOJ.db')
local_student_db_path = os.path.join(ROOT_PATH, 'data/student_code.db')
local_token_code_db = os.path.join(ROOT_PATH, 'data/token_code.db')
local_test_experiment_db = os.path.join(ROOT_PATH, 'data/test_experiment.db')
local_test_experiment_finish_db = os.path.join(ROOT_PATH, 'data/test_experiment_finish.db')
local_c_test_experiment_db = os.path.join(ROOT_PATH, 'data/test_c_experiment.db')
local_c_test_experiment_finish_db = os.path.join(ROOT_PATH, 'data/test_c_experiment_finish.db')
DEEPFIX_DB = os.path.join(ROOT_PATH, 'data/deepfix.db')

spider_ip = '218.94.159.108'
spider_port = 22

cache_data_path = os.path.join(ROOT_PATH, 'data')

cpp_tmp_dir = '/tmp/program_ai/'
cpp_tmp_filename = 'main.cpp'
cpp_tmp_path = os.path.join(cpp_tmp_dir, cpp_tmp_filename)

char_sign_dict = {
    " ": 0, "!": 1, "\"": 2, "#": 3, "$": 4, "%": 5, "&": 6, "'": 7, "(": 8, ")": 9,
    "*": 10, "+": 11, ",": 12, "-": 13, ".": 14, "/": 15, "0": 16, "1": 17, "2": 18, "3": 19,
    "4": 20, "5": 21, "6": 22, "7": 23, "8": 24, "9": 25, ":": 26, ";": 27, "<": 28, "=": 29,
    ">": 30, "?": 31, "@": 32, "A": 33, "B": 34, "C": 35, "D": 36, "E": 37, "F": 38, "G": 39,
    "H": 40, "I": 41, "J": 42, "K": 43, "L": 44, "M": 45, "N": 46, "O": 47, "P": 48, "Q": 49,
    "R": 50, "S": 51, "T": 52, "U": 53, "V": 54, "W": 55, "X": 56, "Y": 57, "Z": 58, "[": 59,
    "\\": 60, "]": 61, "^": 62, "_": 63, "`": 64, "a": 65, "b": 66, "c": 67, "d": 68, "e": 69,
    "f": 70, "g": 71, "h": 72, "i": 73, "j": 74, "k": 75, "l": 76, "m": 77, "n": 78, "o": 79,
    "p": 80, "q": 81, "r": 82, "s": 83, "t": 84, "u": 85, "v": 86, "w": 87, "x": 88, "y": 89,
    "z": 90, "{": 91, "|": 92, "}": 93, "~": 94, "\n": 95, 'plh': 96,
}

sign_char_dict = {
    0: ' ', 1: '!', 2: '"', 3: '#', 4: '$', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')',
    10: '*', 11: '+', 12: ',', 13: '-', 14: '.', 15: '/', 16: '0', 17: '1', 18: '2', 19: '3',
    20: '4', 21: '5', 22: '6', 23: '7', 24: '8', 25: '9', 26: ':', 27: ';', 28: '<', 29: '=',
    30: '>', 31: '?', 32: '@', 33: 'A', 34: 'B', 35: 'C', 36: 'D', 37: 'E', 38: 'F', 39: 'G',
    40: 'H', 41: 'I', 42: 'J', 43: 'K', 44: 'L', 45: 'M', 46: 'N', 47: 'O', 48: 'P', 49: 'Q',
    50: 'R', 51: 'S', 52: 'T', 53: 'U', 54: 'V', 55: 'W', 56: 'X', 57: 'Y', 58: 'Z', 59: '[',
    60: '\\', 61: ']', 62: '^', 63: '_', 64: '`', 65: 'a', 66: 'b', 67: 'c', 68: 'd', 69: 'e',
    70: 'f', 71: 'g', 72: 'h', 73: 'i', 74: 'j', 75: 'k', 76: 'l', 77: 'm', 78: 'n', 79: 'o',
    80: 'p', 81: 'q', 82: 'r', 83: 's', 84: 't', 85: 'u', 86: 'v', 87: 'w', 88: 'x', 89: 'y',
    90: 'z', 91: '{', 92: '|', 93: '}', 94: '~', 95: '\n', 96: 'plh'
}



standard_library_header_file = {
    "array",
    "deque",
    "forward_list",
    "list",
    "map",
    "queue",
    "set",
    "stack",
    "unordered_map",
    "unordered_set",
    "vector",
    "cassert",
    "cctype",
    "cerrno",
    "cfenv",
    "cfloat",
    "cinttypes",
    "ciso646",
    "climits",
    "clocale",
    "cmath",
    "csetjmp",
    "csignal",
    "cstdarg",
    "cstdbool",
    "cstddef",
    "cstdint",
    "cstdio",
    "cstring",
    "ctgmath",
    "ctime",
    "cuchar",
    "cwchar",
    "cwctype",
    "fstream",
    "iomanip",
    "ios",
    "iosfwd",
    "iostream",
    "istream",
    "ostream",
    "sstream",
    "streambuf",
    "algorithm",
    "bitset",
    "chrono",
    "codecvt",
    "complex",
    "exception",
    "functional",
    "initializer_list",
    "iterator",
    "limits",
    "locale",
    "memory",
    "new",
    "numeric",
    "random",
    "ratio",
    "regex",
    "stdexcept",
    "string",
    "system_error",
    "tuple",
    "typeindex",
    "typeinfo",
    "type_traits",
    "utility",
    "valarray"
}

standard_library_defined_identifier = {
    "all_of",
    "any_of",
    "adjacent_find",
    "binary_search",
    "copy",
    "copy_backward",
    "copy_if",
    "copy_n",
    "count",
    "count_if",
    "equal",
    "equal_range",
    "fill",
    "fill_n",
    "find",
    "find_end",
    "find_first_of",
    "find_if",
    "find_if_not",
    "for_each",
    "generate",
    "generate_n",
    "includes",
    "inplace_merge",
    "is_heap",
    "is_heap_until",
    "is_partitioned",
    "is_permutation",
    "is_sorted",
    "is_sorted_until",
    "iter_swap",
    "lexicographical_compare",
    "lower_bound",
    "make_heap",
    "max",
    "max_element",
    "merge",
    "min",
    "minmax",
    "minmax_element",
    "min_element",
    "mismatch",
    "move",
    "move_backward",
    "next_permutation",
    "none_of",
    "nth_element",
    "partial_sort",
    "partial_sort_copy",
    "partition",
    "partition_copy",
    "partition_point",
    "pop_heap",
    "prev_permutation",
    "push_heap",
    "random_shuffle",
    "remove",
    "remove_copy",
    "remove_copy_if",
    "remove_if",
    "replace",
    "replace_copy",
    "replace_copy_if",
    "replace_if",
    "reverse",
    "reverse_copy",
    "rotate",
    "rotate_copy",
    "search",
    "search_n",
    "set_difference",
    "set_intersection",
    "set_symmetric_difference",
    "set_union",
    "shuffle",
    "sort",
    "sort_heap",
    "stable_partition",
    "stable_sort",
    "swap",
    "swap_ranges",
    "transform",
    "unique",
    "unique_copy",
    "upper_bound",
    "cin",
    "cout",
    "cerr",
    "clog",
    "wcin",
    "wcout",
    "wcerr",
    "wclog",
    "string",
    "stoi",
    "stol",
    "stoul",
    "stoll",
    "stoull",
    "stof",
    "stod",
    "stold",
    "begin",
    "end",
    "append",
    "assign",
    "at",
    "back",
    "capacity",
    "cbegin",
    "cend",
    "clear",
    "compare",
    "copy",
    "crbegin",
    "crend",
    "c_str",
    "data",
    "empty",
    "end",
    "erase",
    "find",
    "find_first_not_of",
    "find_first_of",
    "find_last_not_of",
    "find_last_of",
    "front",
    "get_allocator",
    "insert",
    "length",
    "max_size",
    "pop_back",
    "push_back",
    "rbegin",
    "rend",
    "replace",
    "reserve",
    "resize",
    "rfind",
    "std",
    "shrink_to_fit",
    "size",
    "substr",
    "swap",
    "npos",
    "getline",
    "vector",
    "system",
    "pow",
    "main"
}

pre_defined_cpp_token = set(keywords.keys()) | set(operators.values()) | standard_library_defined_identifier

keyword_weight_dict = {
 '(': 10,
 ')': 10,
 '{': 10,
 '}': 10,
 ';': 10,
 ',': 10,
 ')': 10,
 ')': 10,
}


# ---------------------------------------- pre diefined c token --------------------------------#

c_standard_library_defined_identifier = {
    'erfc',
    'nexttoward',
    'fflush',
    'lrintf',
    'rename',
    'nearbyintf',
    'atanhl',
    'logl',
    'vsprintf',
    'strncmp',
    'cosh',
    'truncf',
    'ldexpf',
    'putchar',
    'erf',
    'malloc',
    'puts',
    'powf',
    'roundl',
    'atan',
    'strcpy',
    'roundf',
    'round',
    'fprintf',
    'modff',
    'floor',
    'logbf',
    'freopen',
    'nextafter',
    'fmin',
    'strtof',
    'memset',
    'nexttowardf',
    'fputs',
    'putc',
    'ftell',
    'llrintl',
    'lround',
    'acoshl',
    'sqrtf',
    'stdin',
    'wcstombs',
    'atanf',
    'nextafterl',
    'fgets',
    'strxfrm',
    'erfcl',
    'strpbrk',
    'fread',
    'clearerr',
    'scalblnf',
    'abs',
    'atan2f',
    'exp2',
    'atol',
    'atan2',
    'strtol',
    'rint',
    'floorf',
    'atoi',
    'rintl',
    'fminl',
    'lgammaf',
    'logbl',
    'lgamma',
    'fmax',
    'scalblnl',
    'wctomb',
    'strrchr',
    'erff',
    'vsscanf',
    'nearbyint',
    'ilogbf',
    'fabsl',
    'sqrt',
    'scanf',
    'scalbln',
    'tgammal',
    'sscanf',
    'srand',
    'strcspn',
    'nexttowardl',
    'acosf',
    'log10l',
    'vprintf',
    'memcpy',
    'calloc',
    'expm1f',
    'labs',
    'scalbn',
    'strspn',
    'atof',
    'frexp',
    'logf',
    'cos',
    'rintf',
    'nanf',
    'vsnprintf',
    'ldexp',
    'qsort',
    'cosf',
    '__codecvt_noconv',
    'expm1l',
    'ferror',
    'log2',
    'truncl',
    'sinf',
    'nextafterf',
    'lroundl',
    'copysignl',
    'gets',
    'acosl',
    'tanf',
    'fopen',
    'fclose',
    'remainderf',
    'log2f',
    'strcoll',
    'fwrite',
    'strerror',
    'exp2f',
    'tanhf',
    'copysignf',
    'llround',
    'mbtowc',
    'feof',
    'fmodl',
    'remquo',
    'llroundf',
    'lgammal',
    'exp2l',
    'erfcf',
    'fscanf',
    'strstr',
    'tanhl',
    'sprintf',
    'sinh',
    'tgammaf',
    'asinl',
    'strtoll',
    'realloc',
    'llrintf',
    'strncat',
    'nan',
    'ceilf',
    'frexpl',
    'setbuf',
    'fma',
    'tanh',
    'getc',
    'atanhf',
    'abort',
    'getchar',
    'fsetpos',
    'copysign',
    'sinhl',
    'asinf',
    'nanl',
    'llrint',
    'memcmp',
    'atan2l',
    'strchr',
    'log',
    'trunc',
    'ilogb',
    'log2l',
    'hypot',
    'system',
    'ceil',
    'printf',
    'acos',
    'vfprintf',
    'scalbnl',
    'strtok',
    'logb',
    'lrint',
    'sqrtl',
    'exp',
    'nearbyintl',
    'atexit',
    'rand',
    'fseek',
    'atanl',
    'setvbuf',
    'strcmp',
    'fmod',
    'remove',
    'free',
    'acoshf',
    'erfl',
    'tmpnam',
    'bsearch',
    'remquof',
    'tanl',
    'coshl',
    'fdim',
    'atanh',
    'asin',
    'pow',
    'expm1',
    'strncpy',
    'fdimf',
    'tmpfile',
    'strtod',
    'tan',
    'fmaxf',
    'cosl',
    'llroundl',
    'asinh',
    'fmaxl',
    'fabs',
    'vfscanf',
    'strtoul',
    'log1pl',
    'memchr',
    'remainder',
    'stdout',
    'hypotf',
    'sinhf',
    'mbstowcs',
    'vscanf',
    'asinhf',
    'strlen',
    'expf',
    'strtoull',
    'remquol',
    'ilogbl',
    'perror',
    'cbrtf',
    'fdiml',
    'log10',
    'ldiv',
    'ungetc',
    'log10f',
    'cbrt',
    'sinl',
    'rewind',
    'fmaf',
    'frexpf',
    'acosh',
    'asinhl',
    'fmodf',
    'strcat',
    'div',
    'remainderl',
    'log1p',
    'floorl',
    'modfl',
    'hypotl',
    'fmal',
    'fputc',
    'fminf',
    'exit',
    'fabsf',
    'sin',
    'powl',
    'atoll',
    'mblen',
    'modf',
    'expl',
    'strtold',
    'cbrtl',
    'memmove',
    'stderr',
    'snprintf',
    'llabs',
    'lldiv',
    'lroundf',
    'log1pf',
    'tgamma',
    'coshf',
    'getenv',
    'fgetpos',
    'ceill',
    'lrintl',
    'fgetc',
    'scalbnf'
}

c_standard_library_defined_types = {
    '_IO_FILE',
    '__mbstate_t',
    'FILE',
    'float_t',
    '__off_t',
    '__ssize_t',
    'lldiv_t',
    '__compar_fn_t',
    'wchar_t',
    'double_t',
    '__gnuc_va_list',
    '__off64_t',
    'fpos_t',
    '_G_fpos_t',
    'size_t',
    '_IO_lock_t',
    'ldiv_t',
    'div_t'
}

keywords = (
    '_BOOL', '_COMPLEX', 'AUTO', 'BREAK', 'CASE', 'CHAR', 'CONST',
    'CONTINUE', 'DEFAULT', 'DO', 'DOUBLE', 'ELSE', 'ENUM', 'EXTERN',
    'FLOAT', 'FOR', 'GOTO', 'IF', 'INLINE', 'INT', 'LONG',
    'REGISTER', 'OFFSETOF',
    'RESTRICT', 'RETURN', 'SHORT', 'SIGNED', 'SIZEOF', 'STATIC', 'STRUCT',
    'SWITCH', 'TYPEDEF', 'UNION', 'UNSIGNED', 'VOID',
    'VOLATILE', 'WHILE', '__INT128',
)

keyword_map = {}
for keyword in keywords:
    if keyword == '_BOOL':
        keyword_map['_Bool'] = keyword
    elif keyword == '_COMPLEX':
        keyword_map['_Complex'] = keyword
    else:
        keyword_map[keyword.lower()] = keyword

keyword_map = reverse_dict(keyword_map)

operator_map = {
    'PLUS': '+',
    'MINUS': '-',
    'TIMES': '*',
    'DIVIDE': '/',
    'MOD': '%',
    'OR': '|',
    'AND': '&',
    'NOT': '~',
    'XOR': '^',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'LOR': '||',
    'LAND': '&&',
    'LNOT': '!',
    'LT': '<',
    'GT': '>',
    'LE': '<=',
    'GE': '>=',
    'EQ': '==',
    'NE': '!=',

    # Assignment operators
    'EQUALS': '=',
    'TIMESEQUAL': '*=',
    'DIVEQUAL': '/=',
    'MODEQUAL': '%=',
    'PLUSEQUAL': '+=',
    'MINUSEQUAL': '-=',
    'LSHIFTEQUAL': '<<=',
    'RSHIFTEQUAL': '>>=',
    'ANDEQUAL': '&=',
    'OREQUAL': '|=',
    'XOREQUAL': '^=',

    # Increment/decrement
    'PLUSPLUS': '++',
    'MINUSMINUS': '--',

    # ->
    'ARROW': '->',

    # ?
    'CONDOP': '?',

    # Delimeters
    'LPAREN': '(',
    'RPAREN': ')',
    'LBRACKET': '[',
    'RBRACKET': ']',
    'COMMA': ',',
    'PERIOD': '.',
    'SEMI': ';',
    'COLON': ':',
    'ELLIPSIS': '...',

    'LBRACE': '{',
    'RBRACE': '}',
}

pre_defined_c_tokens = set(keyword_map.values()) | set(operator_map.values()) | c_standard_library_defined_identifier\
                       | c_standard_library_defined_types


# pre_defined_cpp_token = {'!',
#  '!=',
#  '#',
#  '%',
#  '%=',
#  '&',
#  '&&',
#  '&=',
#  '*',
#  '*=',
#  '+',
#  '++',
#  '+=',
#  ',',
#  '-',
#  '--',
#  '-=',
#  '->',
#  '->*',
#  '.',
#  '.*',
#  '/',
#  '/=',
#  ':',
#  '::',
#  '<',
#  '<<',
#  '<<=',
#  '<=',
#  '=',
#  '==',
#  '>',
#  '>=',
#  '>>',
#  '>>=',
#  '?',
#  'BOOL',
#  'CHAR',
#  'DEC',
#  'IDENT',
#  'LITERAL',
#  'REF',
#  'STRING',
#  'USE',
#  '[',
#  ']',
#  '^',
#  '^=',
#  '__func__',
#  'alignas',
#  'alignof',
#  'and',
#  'and_eq',
#  'asm',
#  'auto',
#  'bitand',
#  'bitor',
#  'bool',
#  'break',
#  'case',
#  'catch',
#  'char',
#  'cin',
#  'class',
#  'compl',
#  'const',
#  'const_cast',
#  'constexpr',
#  'continue',
#  'cout',
#  'decltype',
#  'default',
#  'delete',
#  'do',
#  'double',
#  'else',
#  'enum',
#  'explicit',
#  'export',
#  'extern',
#  'false',
#  'float',
#  'for',
#  'friend',
#  'goto',
#  'if',
#  'inline',
#  'int',
#  'long',
#  'mutable',
#  'namespace',
#  'nan',
#  'new',
#  'noexcept',
#  'not',
#  'not_eq',
#  'nullptr',
#  'operator',
#  'or',
#  'or_eq',
#  'pair',
#  'private',
#  'protected',
#  'public',
#  'register',
#  'reinterpret_cast',
#  'return',
#  'short',
#  'signed',
#  'size_t',
#  'sizeof',
#  'static',
#  'static_cast',
#  'std',
#  'string',
#  'struct',
#  'switch',
#  'template',
#  'this',
#  'thread_local',
#  'throw',
#  'true',
#  'try',
#  'typedef',
#  'typeid',
#  'typename',
#  'union',
#  'unsigned',
#  'using',
#  'vector',
#  'virtual',
#  'void',
#  'volatile',
#  'wchar_t',
#  'while',
#  'xor',
#  'xor_eq',
#  '|',
#  '|=',
#  '||',
#  '~'}
