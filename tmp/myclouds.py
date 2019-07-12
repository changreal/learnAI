# -$- coding:utf-8 -*-

"""
@author:Zhang
@file:myclouds.py
@time:2019-07-08 20:38
"""

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import matplotlib.pyplot as plt
from scipy.misc import imread

name = 'nini'

text = open('H:\qqchat/filters/'+name+'.txt', 'r', encoding='utf-8').read()  # 打开过滤好的txt文件
print(text)
bg_pic = imread('H:\qqchat/bg.jpg')     # 导入词云背景
wordcloud = WordCloud(mask=bg_pic, background_color='white', scale=1.5, font_path='C:/Windows/Fonts/simhei.ttf', width=1000,height=600,stopwords={'表情','糊脸','拍桌','拍头'},min_font_size=10,max_font_size=36,font_step=4,
).generate(text)    # 定义词云的各种变量，可以控制词云的形式，这里的控制变量可以去网上查找，stopwords={'表情','糊脸','拍桌','拍头''是为了过滤掉里面的部分表情信息
image_colors = ImageColorGenerator(bg_pic)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud.to_file('H:\qqchat/'+name+'jpg')   # 输出词云