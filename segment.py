import re
import jieba


input_text_path = '../bigdata/3text.txt'
seg_text_path = '../bigdata/seg_3text.txt'
stopwords_path = '../resource/stopwords_new.txt'

# 添加自定义词典
jieba.load_userdict('../resource/legal_instrument_lexicon.txt')
jieba.load_userdict('../resource/THUOCL_law_lexicon.txt')


def read_stops(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 去标点符号、字母数字等
def removePunc(text):
    pattern = r'[\d+\s+\.\!\/_,?=$%^)*(+\"\']+|[+——！:；>    ，。？、~@#￥%……&*（）]'
    newstr = re.sub(pattern, '', text)
    r1 = re.sub("[<>]", "", newstr)
    r2 = re.sub("[a-zA-Z0-9]", "", r1)
    return r2


# 分词
def seg_words(filepath):
    stopwords = read_stops(stopwords_path)
    file_in = open(filepath, 'r', encoding='utf-8')
    line = file_in.readline()
    count = 1
    file_seg = open(seg_text_path, 'w', encoding='utf-8')
    while line:
        # seg = line.strip('\r\n').split('\t')
        # anyou = seg[1]  # 案由
        # focus = seg[3]  # 争议焦点
        # if anyou in anyous:
        words = jieba.cut(line, cut_all=False)  # 分词
        new_weords = []
        for word in words:
            if word not in stopwords:  # 去停用词
                new_word = removePunc(word)
                if len(new_word) >= 2:
                    new_weords.append(new_word)
        if new_weords:
            file_seg.write('\t'.join(new_weords) + '\n')
        if count % 100 == 0:
            print('have processed %d lines.' % (count))
        count += 1
        line = file_in.readline()
    file_in.close()
    file_seg.flush()
    file_seg.close()


if __name__ == '__main__':
    print('start processing...')
    seg_words(input_text_path)
    print('finished!')
