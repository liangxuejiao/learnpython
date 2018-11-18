import re


def extract_names(file_in_path, file_out_path):
    anyous = ['人格权纠纷', '姓名权纠纷', '肖像权纠纷', '名誉权纠纷', '荣誉权纠纷', '隐私权纠纷', '人身自由权纠纷', '一般人格权纠纷']
    file_in = open(file_in_path, 'r', encoding='utf-8')
    file_out = open(file_out_path, 'w', encoding='utf-8')
    line0 = file_in.readline()
    count = 1
    s = 'anyoucode' + '\t' + 'anyou' + '\t' + 'keyid' + '\t' + 'label' + '\t' + 'name' + '\n'
    while line0:
        line = line0.strip()
        line = line.split('\t')
        if line[1] in anyous and line[3] == '3':
            name1 = re.findall(r"^(原告：|原告:|原告)(.+?)(诉|，|,|。|；|、|\))", line[-2])
            name2 = re.findall(r"^(被告：|被告:|被告)(.+?)(，|,|。|；|、|\))", line[-2])
            name0 = re.findall(r"^(原告\(反诉被告\)：|原告\(反诉被告\))(.+?)，", line[-2])
            name00 = re.findall(r"^(被告\(反诉原告\)：|被告\(反诉原告\))(.+?)，", line[-2])
            # print(line[-2])
            if name0:
                if name0[0][1] not in s:
                    s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '原告' + '\t' + name0[0][1] + '\n'
            elif name00:
                if name00[0][1] not in s:
                    s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '被告' + '\t' + name00[0][1] + '\n'
            elif name1:
                name1_empty = re.findall(r"委托|代理", name1[0][1])
                name1_new = re.findall(r"(.+?)(\()", name1[0][1])
                if not name1_empty:
                    if name1_new:
                        if name1_new[0][0] not in s:
                            s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '原告' + '\t' + name1_new[0][0] + '\n'
                    elif name1[0][1] not in s:
                        s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '原告' + '\t' + name1[0][1] + '\n'
            elif name2:
                name2_empty = re.findall(r"委托|代理", name2[0][1])
                name2_another = re.findall(r"(以下简称“|以下简称)(.+?)($|”)", name2[0][1])
                name2_new = re.findall(r"(.+?)(\()", name2[0][1])
                if not name2_empty:
                    if name2_new:
                        if name2_new[0][0] not in s:
                            s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '被告' + '\t' + name2_new[0][0] + '\n'
                    elif name2[0][1] not in s:
                        s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '被告' + '\t' + name2[0][1] + '\n'
                    if name2_another:
                        s += line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + '被告' + '\t' + name2_another[0][1] + '\n'
        if count % 100000 == 0:
            print('Have processed %d lines.' % (count))
        count += 1
        line0 = file_in.readline()
    file_in.close()
    file_out.write(s)
    file_out.flush()
    file_out.close()


if __name__ == '__main__':
    filein_path = '../data/m1_section_utf8.txt'
    fileout_path = '../data/m1_names.txt'
    extract_names(filein_path, fileout_path)
