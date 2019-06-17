import pandas as pd

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s')
import Congfig

def read_file(sep, filename):
    """
    读取csv，tsv文件
    :param sep: 分隔符
    :param filename: 文件名
    :return: []
    """
    logging.debug("开始读取csv/tsv文件")
    train = pd.read_csv(filename, sep=sep, header=0, encoding="utf-8", engine='python')
    v = train.values
    data_list = []
    for _ in v:
        data_list.append(_[0])
    logging.debug("结束读取csv/tsv文件，一共读取{}条数据".format(len(data_list)))
    return data_list


def _write_file(filename, contents):
    """
    写文件
    :param filename: 文件路径
    :param contents: 内容[]列表
    :return:
    """
    logging.debug("写入文件中，文件名：{}, 一共{}条数据".format(filename, len(contents)))
    with open(filename, 'w') as f:
        for content in contents:
            f.write(content + "\n")
    logging.debug("写入结束")

def _read_file(filename):
    """
    按行读取文件
    :param filename: 文件路径
    :return: []列表数据
    """
    logging.debug("开始读取文件")
    with open(filename, 'r') as f:
        data_list = f.readlines()
    logging.debug("读取结束")
    return data_list


def sepFile(contents, n):
    """
    切割数据，将数据分为n份然后保存成test.goods.txt
    :param contents:
    :return:
    """
    logging.debug("开始切割数据，并写入文件")
    data_len = len(contents)
    step_size = int(data_len / n)
    # print(step_size, data_len)
    data_lists = []
    for i in range(1, n+1):
        if i == n :
            _ = contents[step_size * (i - 1):]
        else:
            _ = contents[step_size * (i - 1):step_size * i]

        data_lists.append(_)
    logging.debug("切割结束，一共切割为{}份".format(n))
    # 写入文件
    # for addr in Congfig.SERVER_ADDRESS:
    filename = Congfig.BASE_PATH + "test.goods.txt"
    # filename = "textcnn_classify/dataset/asks/goods/test{}.goods.txt".format(i+1)
    _write_file(filename, data_lists)


if __name__ == '__main__':
    # 读取
    data_list = read_file(r"\t", "../dataset/asks/goods/test.tsv")
    # 切割
    sepFile(data_list, 10)
