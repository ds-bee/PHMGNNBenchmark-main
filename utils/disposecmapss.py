import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

def disposeCmapss():
    train_df = pd.read_csv('D:\\实验\\CMAPSS\\CMAPSSDataNASA数据集六\\train_FD001.txt', sep=" ", header=None)  # train_dr.shape=(20631, 28)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # 去掉26,27列并用新生成的数组替换原数组
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                        's18', 's19', 's20', 's21']

    # 先按照'id'列的元素进行排序，当'id'列的元素相同时按照'cycle'列进行排序
    train_df = train_df.sort_values(['id', 'cycle'])

    test_df = pd.read_csv('D:\\实验\\CMAPSS\\CMAPSSDataNASA数据集六\\test_FD001.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
                       's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                       's18', 's19', 's20', 's21']


    truth_df = pd.read_csv('D:\\实验\\CMAPSS\\CMAPSSDataNASA数据集六\\RUL_FD001.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

    """Data Labeling - generate column RUL"""
    # 按照'id'来进行分组，并求出每个组里面'cycle'的最大值,此时它的索引列将变为id
    # 所以用reset_index()将索引列还原为最初的索引
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    # 将rul通过'id'合并到train_df上，即在相同'id'时将rul里的max值附在train_df的最后一列
    train_df = train_df.merge(rul, on=['id'], how='left')
    # 加一列，列名为'RUL'
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    # 将'max'这一列从train_df中去掉
    train_df.drop('max', axis=1, inplace=True)

    """
    test_df(13096, 28)

       id  cycle  setting1  setting2  ...       s20       s21  cycle_norm  RUL
    0   1      1  0.632184  0.750000  ...  0.558140  0.661834     0.00000  142
    1   1      2  0.344828  0.250000  ...  0.682171  0.686827     0.00277  141
    2   1      3  0.517241  0.583333  ...  0.728682  0.721348     0.00554  140
    3   1      4  0.741379  0.500000  ...  0.666667  0.662110     0.00831  139
    ...
    """

    """pick a large window size of 50 cycles"""
    sequence_length = 50

    def gen_sequence(id_df, seq_length, seq_cols):
        data_array = id_df[seq_cols].values
        num_elements = data_array.shape[0]
        for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
            yield data_array[start:stop, :]

    """pick the feature columns"""
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    '''
    sequence_cols=['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
    's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    '''
    # 下一行所用的gen_sequence()中第一个参数是训练集中id为1的部分，第二个参数是50, 第三个参数如下所示
    val = list(gen_sequence(train_df[train_df['id'] == 1], sequence_length, sequence_cols))
    val_array = np.array(val)  # val_array.shape=(142, 50, 25)  142=192-50

    '''
    sequence_length= 50
    sequence_cols= ['setting1', 'setting2', 'setting3', 'cycle_norm', 's1', 's2', 's3', 's4', 's5', 's6', 
    's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

    train_df[train_df['id'] == 1]=

    id  cycle  setting1  setting2  ...       s20       s21  RUL  cycle_norm
    0     1      1  0.459770  0.166667  ...  0.713178  0.724662  191    0.000000
    1     1      2  0.609195  0.250000  ...  0.666667  0.731014  190    0.002770
    2     1      3  0.252874  0.750000  ...  0.627907  0.621375  189    0.005540
    3     1      4  0.540230  0.500000  ...  0.573643  0.662386  188    0.008310
    4     1      5  0.390805  0.333333  ...  0.589147  0.704502  187    0.011080
    ..   ..    ...       ...       ...  ...       ...       ...  ...         ...
    187   1    188  0.114943  0.750000  ...  0.286822  0.089202    4    0.518006
    188   1    189  0.465517  0.666667  ...  0.263566  0.301712    3    0.520776
    189   1    190  0.344828  0.583333  ...  0.271318  0.239299    2    0.523546
    190   1    191  0.500000  0.166667  ...  0.240310  0.324910    1    0.526316
    191   1    192  0.551724  0.500000  ...  0.263566  0.097625    0    0.529086

    [192 rows x 28 columns]
    '''
    # 将每个id对应的训练集转换为一个sequence
    seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
               for id in train_df['id'].unique())

    # 生成sequence并把它转换成np array
    # 在train_FD001.txt中按照id分成了100组数据，对每一组进行sequence后每组会减少window_size的大小
    # 20631-100*50 = 15631
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)  # seq_array.shape=(15631, 50, 25)
    seq_tensor = torch.tensor(seq_array)
    seq_tensor = seq_tensor.view(15631, 1, 50, 25)
    print("seq_tensor_shape=", seq_tensor.shape)
    print(seq_tensor[0].shape)

    """generate labels"""

    def gen_labels(id_df, seq_length, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[seq_length:num_elements, :]

    label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
                 for id in train_df['id'].unique()]

    label_array = np.concatenate(label_gen).astype(np.float32)  # label_array.shape=(15631, 1)
    label_scale = (label_array - np.min(label_array)) / (np.max(label_array) - np.min(label_array))

    label_tensor = torch.tensor(label_scale)
    label_tensor = label_tensor.view(-1)
    print("label=", label_tensor[:142])

    num_sample = len(label_array)
    print("num_sample=", num_sample)
    input_size = seq_array.shape[2]
    hidden_size = 100
    num_layers = 2




if __name__ == '__main__':
    disposeCmapss()