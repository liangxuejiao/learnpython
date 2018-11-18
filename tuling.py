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
    for i in range(1, 100):
        queryStr = input("我:".encode('utf-8'))
        TULINURL = "%s%s" % (raw_TULINURL, urllib.parse.quote(queryStr))
        req = urllib.request.Request(url=TULINURL)
        result = urllib.request.urlopen(req).read()
        hjson = json.loads(result)
        length = len(hjson.keys())
        content = hjson['text']

        if length == 3:
            return 'robots:' + content + hjson['url']
        elif length == 2:
            return 'robots:' + content


if __name__ == '__main__':
    print ("你好，请输入内容:")
    contents = result()
    print (contents)
