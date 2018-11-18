# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import codecs
import csv


class QalistPipeline(object):
    def __init__(self):
    	self.qalist = codecs.open("qalist.csv", "w", encoding="utf-8")

    def process_item(self, item, spider):

    	self.qalist.write("%s\t%s\n" % (item["question"], item["answer"]))
        return item
