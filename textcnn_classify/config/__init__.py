#__author__ = chenzhengqiang
#__date__ = 2018-06-05
#__desc__ = configuration of text classification with deep learning by cnn
import os


class TextCnnConfig(object):

    # model_save_dir = 'checkpoints'
    model_save_dir = 'textcnn_classify/checkpoints'
    model_file_prefix = 'text-cnn-model'
    embedding_dimension = 64 # 词向量纬度
    sequence_length = 200 # 序列长度
    # num_classes = 30 # 分类数量
    num_classes = 1258 # 分类数量
    num_filters = 128
    kernel_size = 5
    vocab_size = 5000
    hidden_dimension = 128
    dropout_keep_prob = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 20# 迭代次数
    save_per_batch = 10
    train_device = "/cpu:0"
    # 学习率递减
    # regularizer_rate = 0.0001
    # lr_decay = 0.99
    # lr_step = 1000

    # allow_growth = True

