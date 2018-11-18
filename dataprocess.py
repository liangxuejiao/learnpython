# -*- coding:utf-8 -*-

from __future__ import print_function

import re
import csv
import jieba
import jieba.posseg
import pandas as pd


class TextCleaner(object):
    """
    A class for cleaning text
    """
    def clean(self, text):
        """
        Clean data
        :param text: the raw string
        :return: the string after cleaning
        """

        # 只保留中文的正则表达式
        cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        text = cop.sub("", text)

        return text.strip()


class Segmentor(object):
    """
    A class for segmenting text
    """
    def __init__(self, user_dict=True):
        """
        Initial
        :param user_dict: whether use user dict
        """
        self.seg = jieba
        self.seg_pos = self.seg.posseg
        if user_dict:
            self.seg.load_userdict("userdict")

    def seg_token(self, text):
        """
        :param text: the raw string
        :return: a list of token
        """
        return self.seg.lcut(text)

    def seg_token_pos(self, text):
        """
        :param text: the raw string
        :return: a list of token/pos
        """
        sentence_seged = self.seg_pos.lcut(text.strip())
        outstr = ''
        for x in sentence_seged:
            outstr += "{}，".format(x.flag)
        return outstr


class PreProcess(object):
    """
    A class for feature selecting and extracting
    """
    def __init__(self, file_path):
        """
        Initial
        :param file_path: the comment data path
        """
        corpus = pd.read_csv(file_path, "\t")
        # print(corpus)
        self.corpus = corpus.rename(columns={"question": "question",
                                             "answer": "answer"})
        # print(self.corpus)
        self.cleaner = TextCleaner()
        self.seg = Segmentor()
        self.segment()

        # 预处理后保存结果
        self.corpus.to_csv("QApos.csv", sep="\t", index=0, encoding="utf-8")

    def segment(self):
        """
        Segment text
        """
        def seg(row):

            s = self.cleaner.clean(row["question"])
            a = self.seg.seg_token_pos(s)
            row["question_seg"] = ["%s" % (pos) for pos in jieba.posseg.lcut(s)]
            print(row["question_seg"])
            # 替换词性
            a = re.sub("vn", "A", a)
            a = re.sub("nz", "B", a)
            a = re.sub("uj", "C", a)
            a = re.sub("nt", "D", a)
            a = re.sub("ng", "E", a)
            a = re.sub("ns", "F", a)
            a = re.sub("nrt", "G", a)
            a = re.sub("nr", "H", a)
            a = re.sub("ul", "J", a)
            a = re.sub("zg", "K", a)
            a = re.sub("ad", "M", a)
            a = re.sub("x", "n", a)
            a = re.sub("eng", "n", a)
            a = re.sub("ug", "UG", a)
            row["question_seg_pos"] = re.sub("，", "", a)
            print(row["question_seg_pos"])

            return row

        self.corpus = self.corpus.apply(seg, axis=1)


class Process(object):
    def __init__(self, file_path):
        corpus = pd.read_csv(file_path, "\t")
        # print(corpus)
        self.corpus = corpus.rename(columns={"question": "question",
                                             "answer": "answer",
                                             "question_seg": "question_seg",
                                             "question_seg_pos": "question_seg_pos"})

    # def seg(row):


if __name__ == "__main__":
    file_name = "output"
    pre = PreProcess(file_name)
