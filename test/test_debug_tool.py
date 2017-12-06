from common.debug_tool import *

if __name__ == '__main__':
    print("memory_size:{}".format(memory()))
    l = [9] * 1000000000
    print("memory_size:{}".format(memory()))
