from opencc import OpenCC

def zh_ch_transform():
    '''
        轉換 cmn.txt 簡體字 -> 繁體字
        輸出到 cmn_zh_tw.txt
        Only run once
    '''
    with open('cmn.txt') as fp:
        lines = fp.readlines()

    newLines = []
    cc = OpenCC('s2t')
    for line in lines:
        e, simple_c, _ = line.split('\t')
        trandition_c = cc.convert(simple_c)
        newLines.append("{}\t{}".format(e, trandition_c))

    with open("cmn_zh_tw.txt", 'w') as fp:
        for line in newLines:
            fp.write("{}\n".format(line))

zh_ch_transform()