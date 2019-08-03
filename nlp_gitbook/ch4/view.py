import pandas as pd
import numpy as np
import jieba
import networkx as nx
import matplotlib.pyplot as plt

# 读取文件
dir = "./"
columns = ['std_id','class','name','classroom','label_1','label_2','label_3','label_4','time','label_5']
df = pd.read_csv(dir + "nd_course_schedule_info.csv",sep='\t',names=columns)

#提取关键列
classes = df['class'].values.tolist()
classrooms = df['classroom'].values.tolist()

# 生成不重复的节点与边
nodes = list(set(classes + classrooms))
weights = [(df.loc[index,'class'],df.loc[index,'classroom'])for index in df.index]
weights =  list(set(weights))

# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False
colors = ['red', 'green', 'blue', 'yellow']

#开始画图
DG = nx.DiGraph() #有向图
DG.add_nodes_from(nodes) #一次性添加多节点，输入的格式为列表
DG.add_edges_from(weights) #添加边，数据格式为列表
nx.draw(DG,with_labels=True, node_size=1000, node_color = colors)
plt.show()
