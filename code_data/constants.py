import os

ROOT_PATH = '/home/lf/Project/program_ai'
DATABASE_PATH = os.path.join(ROOT_PATH, 'data/train.db')
BACKUP_PATH = os.path.join(ROOT_PATH, 'backup/')
LOG_PATH = os.path.join(ROOT_PATH, 'logs/')
CHECKPOINT_PATH = os.path.join(ROOT_PATH, 'checkpoint/')
EPISODES = 'episodes'
STEP_INFO = 'step_info'

spider_key_path = 'key/id_rsa_spider'
remote_db_path = '/home/lf/DataCrawl/scrapy_OJ/release_data/scrapyOJ.db'
local_db_path = '/home/lf/Project/program_ai/data/scrapyOJ.db'

spider_ip = '218.94.159.108'
spider_port = 22

cache_data_path = os.path.join('data', 'cache_data')

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
