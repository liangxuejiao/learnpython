# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:24:49 2018

@author: 33236
"""

# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
import sys
import json

from imp import reload
reload(sys)


API_KEY = '025e2012d96d43e89a8e6738ef27e4e3'
raw_TULINURL = "http://www.tuling123.com/openapi/api?key=%s&info=" % API_KEY


def result():
        queryStr = input("我：")
        TULINURL = "%s%s" % (raw_TULINURL, urllib.parse.quote(queryStr))
        req = urllib.request.Request(url=TULINURL)
        result = urllib.request.urlopen(req).read()  # 用Request类构建了一个完整的请求，增加了headers等一些信息
        hjson = json.loads(result)  # 请求之后的结果
        length = len(hjson.keys())  # 判断dict属性个数
        content = hjson['text']  # 获取dict中text的内容
        if length == 3:  # 当回答中有额外添加url时，dict中会多加一个属性
            return 'robots:' + content + hjson['url']
        elif length == 2:  # 回答中无url
            return 'robots:' + content


if __name__ == '__main__':
    print ("你好，请输入内容:")
    contents = result()  # 调用
    print (contents)
