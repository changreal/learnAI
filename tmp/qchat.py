# -$- coding:utf-8 -*-

"""
@author:Zhang
@file:qchat.py
@time:2019-07-08 20:34
"""


import jieba

newtext = []
name = 'nini'
# 打开E盘下的聊天记录文件qq.txt
for word in open('H:/qqchat/'+name+'txt', 'r', encoding='utf-8'):
    tmp = word[0:4]
    www = word[0:10]
    if (tmp == "2019" or tmp == "===="or tmp == "2018"):  # 过滤掉聊天记录的时间和qq名称
        continue
    if ('http'in www or 'https' in www or 'www'in www ):
        continue
    tmp = word[0:2]
    if (tmp[0] == '[' or tmp[0] == '/'or tmp[0] == '@'):  # 过滤掉图片和表情，例如[图片]，/滑稽
        continue
    newtext.append(word)
# 将过滤掉图片和表情和时间信息和qq名称剩下的文字重新写入E盘下的q1.txt文件中去
with open('H:\qqchat/filters/'+name+'txt', 'w', encoding='utf-8') as f:
    for i in newtext:
        f.write(i)
 # 打开新生成的聊天记录文件
text = open('H:\qqchat/filters/'+name+'txt', 'r', encoding='utf-8').read()
word_jieba = jieba.cut(text, cut_all=True)
word_split = " ".join(word_jieba)