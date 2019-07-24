import pandas as pd
import numpy as np
from hiveplot import HivePlot
import matplotlib.pyplot as plt
import json
from urllib.request import urlopen, quote
import requests,csv
import pandas as pd #导入这些库后边都要用到

# 读取文件
dir = "./"
columns = ['std_id','sex','year','colloge','city']
df = pd.read_csv(dir + "nd_student_info.csv",sep='\t',names=columns)

# 数据处理
cities_stat = df.groupby(by=['city'])['city'].agg({'count': np.size})  # 分组
cities_stat = cities_stat.reset_index().sort_values(by=['count'], ascending=False) # 排序


#经纬度转换参考https://www.jianshu.com/p/773ff5f08a2c

def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = 'sqGDDvCDEZPSz24bt4b0BpKLnMk1dv6M'
    add = quote(address) #由于本文城市变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + add  + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode() #将其他编码的字符串解码成unicode
    temp = json.loads(res)  #对json数据进行解析
    return temp

df_result = pd.DataFrame(columns=('city','lng','lat','count'))
for indexs in cities_stat.index:
    b = cities_stat.loc[indexs].values[0].strip() #将第一列city读取出来并清除不需要字符
    c= cities_stat.loc[indexs].values[1]  #将第二列price读取出来并清除不需要字符
    try:
        lng = getlnglat(b)['result']['location']['lng'] #采用构造的函数来获取经度
        lat = getlnglat(b)['result']['location']['lat'] #获取纬度
        print(lng)
        print(lat)
        dict = {'city':b,'lng':lng,'lat':lat,'count':c}
        print(dict)
        df_result.loc[df_result.shape[0]+1] = dict
    except:
        continue
df_result.to_csv("heat_map.csv",index=False,encoding='utf-8')

#打开保存的带有经纬度的数据
cities = pd.read_csv(dir + "heat_map.csv",sep=',')

print(cities.head())