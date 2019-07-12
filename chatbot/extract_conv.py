# -*- coding:utf-8 -*-

import re
import pickle
import sys
from tqdm import tqdm


# 若有这些字符串，将其用空格表示，用空格 代替来连接line中的字符串
def make_split(line):
    if re.match(r'.*([，···?!\.,!？])$', ''.join(line)):
        return []

    return [', ']


# 是不是一个有用的句子
def good_line(line):
    # 在输入中找到是否含有a-z、A-Z或0-9这样的大小写英文字母和0-9数字。长度大于两个字符则返回Flase，否则返回True。
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


# 正则表达式规则
def regular(sen):
    sen = re.sub(r'\.{3,100}', '···', sen)
    sen = re.sub(r'···{2,100}', '···', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)

    return sen

# 输入最多只有20个字符，x最低的长度、y最低长度
def main(limit=20, x_limit=3, y_limit=6):
    # 导入自己定义的类
    from word_sequence import WordSequence

    print('extract lines')
    fp = open('dgk_shooter_min.conv', 'r', errors='ignore', encoding='utf-8')  # 只读  忽略 utf-8编码打开
    groups = []  # 后面变为[group, group, group, group]形式，groups是三维
    group = []  # group用来存行,group是二维

    for line in tqdm(fp):  # 传入进度条，以进度条形式显示
        if line.startswith('M '):
            line = line.replace('\n', '')  # 去掉回车
            if '/' in line:
                line = line[2:].split('/')  # M 什/么/事/儿/这/么/急/啊/  ，以斜杆切分
            else:
                line = list(line[2:])  # 从第二个开始只是要去掉空格
            line = line[:-1]  # 去掉最后一个空格  ['邱', '先', '生', '戏', '刚', '一', '看', '完', '信', '就', '来', '啦']

            group.append(list(regular(
                ''.join(line))))  # '邱先生戏刚一看完信就来啦' --> [['邱', '先', '生', '戏', '刚', '一', '看', '完', '信', '就', '来', '啦']]
        else:
            if group:  # group有值的话
                groups.append(group)  # 三维
                group = []  # 每次追加完以后要清空
    if group:  # 有值的话才会加
        groups.append(group)
        group = []

    # 对训练语料问答对的处理，构造Q\A问答句，
    # 假设 a1,a2,a3,三句话  （a1,a2),(a1+a2,a3) ,(a1,a2+a3)
    x_data = []  # 问
    y_data = []  # 答

    for group in tqdm(groups):
        for i, line in enumerate(group):  # 枚举  得到list((0,'wr'), (1,'wqf'))

            # 根据（a1,a2),(a1+a2,a3) ,(a1,a2+a3)这样的组合来构造问答语句

            last_line = None  # 上一行
            if i > 0:  # i>0至少是2行了 0这一行，1这一行
                last_line = group[i - 1]
                if not good_line(last_line):  # 不是一个好句子，就为空
                    last_line = None

            next_line = None  # 下一个行
            if i < len(group) - 1:  # 下一行的边界是len()-1
                next_line = group[i + 1]
                if not good_line(next_line):
                    next_line = None

            next_next_line = None
            if i < len(group) - 2:  # 下下一行的边界是len()-2
                next_next_line = group[i + 2]
                if not good_line(next_next_line):
                    next_next_line = None

            if next_line:  # 下一行存在
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:  # 存在当前最后一行和下一行
                x_data.append(last_line + make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)

    print(len(x_data), len(y_data))

    # 构建问答
    for ask, answer in zip(x_data[:20], y_data[:20]):  # 就放20个字符
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)

    # 生成pkl文件备用
    data = list(zip(x_data, y_data))
    data = [
        (x, y) for x, y in data if limit > len(x) >= x_limit and limit > len(y) >= y_limit
    ]  # 又是一遍过滤

    x_data, y_data = zip(*data)
    ws_input = WordSequence()
    ws_input.fit(x_data + y_data)  # 句子传进去
    print('dump')
    pickle.dump((x_data, y_data), open('chatbot.pkl', 'wb'))

    pickle.dump(ws_input, open('ws.pkl', 'wb'))
    print('done')


if __name__ == '__main__':
    main()
