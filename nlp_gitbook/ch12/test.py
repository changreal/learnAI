from snownlp import SnowNLP

# 测试京东好评数据
text1 = u'本本已收到，体验还是很好，功能方面我不了解，只看外观还是很不错很薄，很轻，也有质感。'
print(SnowNLP(text1).sentiments)  # 0.9999619413786875

# 测试京东中评数据
text2 = u'屏幕分辨率一般，送了个极丑的鼠标。'
print(SnowNLP(text2).sentiments)

# 测试京东差评
text3 = u'很差的一次购物体验，细节做得极差了，还有发热有点严重啊，散热不行，用起来就是烫得厉害，很垃圾！！！'
print(SnowNLP(text3).sentiments)