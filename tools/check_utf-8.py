# 检测编码格式

import chardet

with open('../data/男科5-13000.csv', 'rb') as f:
    result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测

print(result['encoding'])  # 打印检测到的编码
