import os
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as keras
import pandas as pd
import json



def _parse_train_file(filename, label_2_id, separate=':'):
    '''
    :param filename: each line's format like the following:label$content\n
    :return:
    '''

    labels, contents = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                content = line.strip().split(separate)
                label = content[0]
                content = ''.join(content[1:])
                if not label or label not in label_2_id:
                    continue
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except Exception as e:
                print(e)
                pass

    return labels, contents


def generates_vocab_file(train_file, vocab_file, label_2_id, vocab_size=5000):
    '''

    :param train_file:each line's format like the following:label$content\n
    :param vocab_file: the file to write the most common vocab
    :param vocab_size: default 5000
    :return:
    '''

    _, contents = _parse_train_file(train_file, label_2_id)

    all_words = []
    for content in contents:
        all_words.extend(content)

    counter = Counter(all_words)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)
    with open(vocab_file, "w", encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def generates_vocab_map(vocab_file):
    '''
    :param vocab_file:
    :return:
    '''

    with open(vocab_file, "r", encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def obtain_vocab_size(vocab_file):
    '''
    :param vocab_file:
    :return:
    '''
    with open(vocab_file, "r", encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    return len(words)

def generates_label_map(labels):
    '''
    :param config: obtain labels from config file:config/__init__.py
    :return:{label:ID,...}
    '''
    return dict(zip(labels, range(len(labels))))



def obtain_inputs_of_cnn(train_file, word_to_id, label_to_id, max_sequence_length=200):
    '''
    :param train_file:
    :param word_to_id:{word:index,...}
    :param label_to_id:{label:index,...}
    :param max_length:
    :return:
    '''

    labels, contents = _parse_train_file(train_file, label_to_id)
    single_word_indices, label_indices = [], []

    for index in range(len(contents)):
        single_word_indices.append([word_to_id[word] for word in contents[index] if word in word_to_id])
        label_indices.append(label_to_id[labels[index]])

            
    '''
    single_word_indices=[
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    ...
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666]
    ]
    
    label_indices=[1,1,1,2,2,2,3,3,3...128]
    '''
    train_inputs = keras.preprocessing.sequence.pad_sequences(single_word_indices, max_sequence_length)
    train_inputs = np.concatenate([train_inputs, train_inputs])
    label_inputs = keras.utils.to_categorical(label_indices, num_classes=len(label_to_id))
    label_inputs = np.concatenate([label_inputs, label_inputs])
    return train_inputs, label_inputs


def generates_batch(x, y, batch_size=64):
    '''
    :param x: train or valid matrix
    :param y: label matrix
    :param batch_size:
    :return: batch data after shuffle
    '''

    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for index in range(num_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, data_len)
        yield x_shuffle[start:end], y_shuffle[start:end]


def read_tsv(filename, train_filename, test_filename, val_filename, labels_filename):
    """
    读取文件并提取分类和内容,写入train.txt
    :return:
    """
    train = pd.read_csv(filename, sep='\t', header=0)
    # 获取商品数据和类型
    data = train.values

    # 将类型和类型id一一对应
    # with open("../data/goods/label_inormation.txt", "r", encoding="utf-8") as f:
    #     type_dict = eval(f.read().strip())

    # print(type_dict)
    type_list = []
    for i, da in enumerate(data):
        # type_value =da[-1]
        # content = da[0]
        goods_list = da[0].split("\t")
        type_value = goods_list[-1]
        content = goods_list[0]
        # print(da[0])
        type_list.append(da[-1])

    #
        if i < int(len(data) * 0.725):
            with open(train_filename, "a") as f_obj:
                f_obj.write(type_value + ":" + content)
                f_obj.write("\n")
        elif i < int(len(data) * 0.95):
            with open(test_filename, "a") as f_obj:
                f_obj.write(type_value + ":" + content)
                f_obj.write("\n")
        elif i < len(data):
            with open(val_filename, "a") as f_obj:
                f_obj.write(type_value + ":" + content)
                f_obj.write("\n")
    #
    # type_list = list(set(type_list))
    # print(len(type_list))
    # with open(labels_filename, "w", encoding="utf-8") as f:
    #     f.write("[")
    #     for t in type_list:
    #         f.write('"' + t + '"' + ",")
    #     f.write("]")
        # s = json.loads(type_list)
        # json.dump(type_list, f)
        # print(type_value)
        # print(content)
        # break


def _read_file(txt_file):
    """读取txt文件"""
    return open(txt_file, 'rb').readlines()


def _read_label(label_filename):
    with open(label_filename, "r") as f:
        type_dict = eval(f.read().strip())

    return type_dict


def save_file(dir_name, label_dict):
    f_train = open('../dataset/asks/goods/train.txt', 'w', encoding='utf-8')
    f_test = open('../dataset/asks/goods/test.txt', 'w', encoding='utf-8')
    f_val = open('../dataset/asks/goods/val.txt', 'w', encoding='utf-8')
    max_size = 0
    for category in os.listdir(dir_name):
        dirs = category
        cat_file = os.path.join(dir_name, category)
        fp = _read_file(cat_file)
        count = 0
        print("类型数据长度: ", len(fp), "类型id：", category)
        print("训练数据大小：", int(len(fp) * 0.75 ) )
        print("验证数据大小：", int(len(fp) * 0.95 ) - int(len(fp) * 0.75 ))
        print("测试数据大小：", len(fp) - int(len(fp) * 0.95 ))
        for line in fp:
            l = line.decode("utf-8")
            line = line.decode("utf-8")
            line = line.split("\t")
            try:
                category = line[0]
                content = line[1]
                try:
                    category = label_dict.get(int(category))[-1]
                    # category = "--".join(category.split(" ")).strip()
                except Exception as e:
                    # print(e)
                    category = line[1]
                    content = line[0]
                    if "--" not in category:
                        category = label_dict.get(int(category))[-1].stirp()
                        # category = "--".join(category.split(" ")).strip()
                    print("*" * 20)
            except Exception as e:
                print(e)
                print("-"*20)
                print(line)
                category = None
                content = None
            if category and content:
                if max_size < len(content):
                    max_size = len(content)
                if count < int(len(fp) * 0.75 ) :
                    f_train.write(category.strip() + ':' + content.strip() + '\n')
                elif count < int(len(fp) * 0.95 ):
                    f_test.write(category.strip() + ':' + content.strip() + '\n')
                elif count < len(fp):
                    f_val.write(category.strip() + ':' + content.strip() + '\n')
            count += 1

        print('Finished', category)

    f_train.close()
    f_test.close()
    f_val.close()
    print("最长的商品名有：", max_size)



if __name__ == '__main__':

    filename = "../dataset/asks/goods/train.tsv"
    train_filename = "../dataset/asks/goods/train.txt"
    test_filename = "../dataset/asks/goods/test.txt"
    val_filename = "../dataset/asks/goods/val.txt"
    labels_filename = "../dataset/asks/goods/labels.json"
    # read_tsv(filename, train_filename, test_filename, val_filename, labels_filename)
    dir_name = "../dataset/asks/deal_data"
    label_filename = "../dataset/asks/goods/label_inormation.txt"
    # type_dict = _read_label(label_filename)
    # print(type_dict)
    label_list = []
    # for v in type_dict.values():
    #     label = v[1]
    #     label_list.append(label)
    # # 写分类
    # with open(labels_filename, "w", encoding="utf-8") as f:
    #     f.write("[")
    #     for t in label_list:
    #         f.write('"' + t + '"' + ",")
    #     f.write("]")
    # 将数据分为测试，训练，验证三组数据
    # save_file(dir_name, type_dict)

    # with open(labels_filename, "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     s = json.load(f)
    #     print(s)
    #     print(len(lines))
    # print(json.loads(open(labels_filename), encoding='utf-8').read())

    # -----读取每个分类有多少数据
    file_list = os.listdir(dir_name)
    count = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    for file in file_list:
        with open(dir_name + "/" + file, "r") as f:
            lines = f.readlines()
            if len(lines) < 50:
                count += 1
                print("数据量: ", len(lines))
                print("分类id：", file)

            if len(lines) < 100:
                count_1 += 1

            if len(lines) < 30:
                count_2 += 1

            if len(lines) < 10:
                count_3 += 1

            if len(lines) < 20:
                count_4 += 1

            if len(lines) < 15:
                count_5 += 1

    print("一共有{}个分类低于50条数据".format(count))
    print("一共有{}个分类低于100条数据".format(count_1))
    print("一共有{}个分类低于30条数据".format(count_2))
    print("一共有{}个分类低于10条数据".format(count_3))
    print("一共有{}个分类低于20条数据".format(count_4))
    print("一共有{}个分类低于15条数据".format(count_5))