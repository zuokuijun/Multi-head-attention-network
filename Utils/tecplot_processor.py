import numpy as np
import os

#  将tecplot导出的数据做数据清洗，获得每个网格点对应的速度梯度以及压力数据
#  将需要训练的数据从原始tecplot数据中提取出来
def generate_field_point():
     root = "../data/tecplot_data/"  # 原始tecplot文件路径
     fileList = os.listdir(root)
     # 对文件夹下的所有翼型文件进行遍历
     for k in range(len(fileList)):
         name = fileList[k].split('.')
         print("正在生成翼型{}训练数据".format(name[0]))
         # 获取文件的绝对路径,用于打开当前文件并对其做数据提取
         file = os.path.join(root, fileList[k])
         # with 方法用于创建一个临时的运行环境，运行环境中的代码执行完后自动安全退出环境
         label = []
         # 打开单个翼型文件
         with open(file, 'r') as f:
            i = 0
            for line in f.readlines():
                 i += 1
                 if(i<=12 or i>=8371):
                      continue
                 else:
                   label.append(line)
                   # 判断当前行是否存在空值
                   line = line.strip("\n")
                   line_list = line.split()
                   if (len(line_list) < 5 ):
                       print("存在空数值{}".format(i))
                       break

         f1 = open(f'../data/prue_data/{name[0]}.txt', mode='w', encoding='utf8')
         for i in range(len(label)):
             f1.writelines(label[i])
             f1.flush()
         f1.close()



if __name__ == '__main__':
    generate_field_point()