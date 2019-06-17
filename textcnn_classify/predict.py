import csv
import os
import json
import threading

import tensorflow as tf
import tensorflow.contrib.keras as keras
from textcnn_classify.config import TextCnnConfig
from textcnn_classify.models.cnn_model import TextCNN
from textcnn_classify.utils import preprocess as UTILS
from textcnn_classify import Global
import Congfig
import time

import logging

from textcnn_classify.utils.deal_data import _read_file

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class CnnModel(object):

    __instance = None

    def __init__(self, config, word_2_id, labels):

        self.word_2_id = word_2_id
        self.labels = labels

        self.config = config
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=os.path.join(config.model_save_dir, config.model_file_prefix))

    def predict(self, contents, file_obj):
        logging.info("开始打标签")
        rlts_list = []
        for content in contents:
            data = [self.word_2_id[word] for word in content if word in self.word_2_id]
            feed_dict = {
                self.model.input_x: keras.preprocessing.sequence.pad_sequences([data], self.config.sequence_length),
                self.model.keep_prob: 1.0
            }
            predict_label = self.session.run(self.model.predict_label, feed_dict=feed_dict)
            rlt_list = [content.strip(), self.labels[predict_label[0]]]
            rlts_list.append(rlt_list)
        # 将结果数据写入文本
        file_obj.writerows(rlts_list)

    def predict_test(self, contents):
        logging.info("开始打标签")
        rlt = []
        for content in contents:
            if content:
                data = [self.word_2_id[word] for word in content if word in self.word_2_id]
                feed_dict = {
                    self.model.input_x: keras.preprocessing.sequence.pad_sequences([data], self.config.sequence_length),
                    self.model.keep_prob: 1.0
                }
                predict_label = self.session.run(self.model.predict_label, feed_dict=feed_dict)
                rlt.append(self.labels[predict_label[0]])
        return rlt


    @classmethod
    def get_obj(self, config, word_2_id, labels):

        if not self.__instance :

            self.__instance = CnnModel(config, word_2_id, labels)

        return self.__instance



def predict(cnn_model):
    filename = "textcnn_classify/dataset/asks/goods/test.goods.txt"
    data_list = _read_file(filename)
    data_len = len(data_list)
    logging.debug("一共有{}条数据.".format(data_len))
    step_size = int(data_len / 2)
    data_lists = [data_list[step_size * (i - 1):step_size * i] for i in range(1, 3)]
    # 创建csv文件，记录训练情况
    csvfile = open(Congfig.BASE_PATH + Congfig.PREDICT_FILENAME, "w")
    writer = csv.writer(csvfile)
    writer.writerow(["ITEM_NAME", "TYPE"])
    start_time = time.time()
    thread_list = []
    for i, contents in enumerate(data_lists):
        t = threading.Thread(target=cnn_model.predict, args=(contents,writer))
        t.start()
        thread_list.append(t)
    for i in thread_list:
        print(i, "线程开启")
        i.join()

    end_time = time.time()
    total_time = end_time - start_time
    csvfile.close()
    logging.debug("一共花费时间：{}".format(total_time))




def main():
    logging.debug("开始打标签")
    VOCAB_FILE = os.path.join(Global.BASE_DIR, Global.DATA_DIR, Global.VOCAB_FILE)
    labels = json.loads(
        open(os.path.join(Global.BASE_DIR, Global.DATA_DIR, Global.LABELS_JSON_FILE), encoding='utf-8').read())
    LABEL_2_ID = UTILS.generates_label_map(labels)
    word_2_id = UTILS.generates_vocab_map(VOCAB_FILE)

    TEXT_CNN_CONFIG = TextCnnConfig()
    TEXT_CNN_CONFIG.vocab_size = UTILS.obtain_vocab_size(VOCAB_FILE)
    cnn_model = CnnModel.get_obj(TEXT_CNN_CONFIG, word_2_id=word_2_id, labels=labels)

    return cnn_model


if __name__ == '__main__':
    filename = "./dataset/asks/goods/test.goods.txt"
    data_list = _read_file(filename)

    data_len = len(data_list)
    step_size = int(data_len / 2)
    # print(step_size, data_len)
    data_lists = [data_list[step_size * (i - 1):step_size * i] for i in range(1, 3)]

    print(len(data_lists))
    print("*" * 10)
    # cnn_model = main()
    # cnn_model.predict("cag")
    cnn_model = main()
    start_time = time.time()
    thread_list = []
    for i, contents in enumerate(data_lists):

        t = "t" + str(i)
        t = threading.Thread(target=cnn_model.predict, args=(contents,))
        t.start()
        thread_list.append(t)
    for i in thread_list:
        print(i, "线程开启")
        i.join()

    end_time = time.time()
    print(end_time-start_time)