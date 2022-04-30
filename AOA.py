import math


def AOA(param):
    X = math.cos((param/180)*math.pi)
    Y = math.sin((param/180)*math.pi)
    return X, Y

def cp(Re, Ma):
    # 黏性系数
    u = 0.000017894
    # 特征长度
    l = 1
    R = 287
    T = 300
    k = 1.4
    temp1 = (Re*u)/(Ma*l)
    temp2 = ((R*T)/k)**0.5
    return temp1 * temp2

if __name__ == '__main__':


    X, Y = AOA(10)
    print("X方向大小为：{}".format(X))
    print("Y方向的大小为:{}".format(Y))
    print("返回远场压力:{}".format(cp(1000, 0.2)))
    # 1800
    # 2000
    # 1200
    # 1000
