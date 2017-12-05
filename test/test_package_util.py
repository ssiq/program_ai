import numpy as np
import math


def almost_equal_array(x, y):
    # print("x:{}, y:{}".format(x, y))
    a1 = np.all((x==math.inf)==(y==math.inf))
    a2 = np.all((x==-math.inf)==(y==-math.inf))
    not_inf = lambda t: (t!=math.inf)&(t!=-math.inf)
    a3 = np.all(np.abs(x[not_inf(x)]-y[not_inf(y)]) < 1e-5)
    # print("a1:{}, a2:{}, a3:{}".format(a1, a2, a3))
    return a1 and a2 and a3