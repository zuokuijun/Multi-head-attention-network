
import   numpy as  np
import  matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def  hist():
    pre = np.loadtxt("./naca0024_pre.txt")
    truth = np.loadtxt("../data/CNN-FCN/test_labels/naca0024.txt")
    res = []
    for i  in range(len(pre)):
        temp = truth[i] - pre[i]
        res.append(temp)

    plt.hist(x=res,
             bins=20,
             color='cornflowerblue',
             edgecolor='black')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

if __name__ == '__main__':
    hist()