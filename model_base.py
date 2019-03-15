import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np
import math
from format_data import load_track
import sys

input_depth = 16
input_height = 128
input_weight = 128

label_legth = 8
level_height = 3

reg_rate = 1e-2
learning_rate = 0.01
moving_avg_rate = 0.9

EPOCHES = 100

class TrainData:
    def __init__(self):
        self._index = 0
        self.batch_size = 128
        self._num_examples = 0
        self._train_data = ""
        self._train_lable = ""

        self._test_data = ""
        self._test_lable = ""
        self._max = 240

        self._test_index = 0
        self._test_num = 0

    def load_data(self, file, file_label):

        train_data = np.load(file)
        train_label = np.load(file_label)

        print(train_data.shape)
        #test
        self._train_lable = train_label[0:self._max]
        self._train_data = train_data[0:self._max]

        self._test_lable = train_label[self._max:]
        self._test_data = train_data[self._max:]

        print self._train_data.shape
        print self._train_lable.shape
        print self._test_data.shape
        print self._test_lable.shape

        if (self._train_lable.shape[0] != self._train_data.shape[0]):
            print "data error!"
            return False
        self._num_examples = self._train_lable.shape[0]
        self._test_num = self._test_lable.shape[0]

    def next_batch(self, batch_size):
        data = self._train_data[self._index: min(self._index + batch_size, self._num_examples)]
        label = self._train_lable[self._index: min(self._index + batch_size, self._num_examples)]

        self._index += batch_size
        if self._index >= self._num_examples:
            self._index = 0
        return data, label

    def get_data_num(self):
        return self._num_examples

    def get_test_data_num(self):
        return self._test_num

    def next_test_batch(self, batch_size):
        data = self._test_data[self._test_index: min(self._test_index + batch_size, self._test_num)]
        lable = self._test_lable[self._test_index: min(self._test_index + batch_size, self._test_num)]

        self._test_index += batch_size
        if self._test_index >= self._test_num:
            self._test_index = 0
        return data, lable


class TrainBigData:
    def __init__(self):
        self._index = 0
        self.batch_size = 128
        self._num_examples = 0
        self._train_data = ""
        self._train_lable = ""

        self._test_data = ""
        self._test_lable = ""
        self._max = 6400

        self._test_index = 0
        self._test_num = 0

    def load_data(self, file=None, file_label=None, train_data= None, train_label=None):

        if file and file_label:
            train_data = np.load(file)
            train_label = np.load(file_label)

        print(len(track_paths))

        self._train_lable = train_label[0:self._max]
        self._train_data = train_data[0:self._max]

        self._test_lable = train_label[self._max:]
        self._test_data = train_data[self._max:]

        print(len(self._train_lable))
        print(len(self._test_lable))

        self._num_examples = len(self._train_lable)
        self._test_num = len(self._test_lable)

    def next_batch(self, batch_size):
        data = self._train_data[self._index: min(self._index + batch_size, self._num_examples)]
        label = self._train_lable[self._index: min(self._index + batch_size, self._num_examples)]

        self._index += batch_size
        if self._index >= self._num_examples:
            self._index = 0

        new_data = []
        new_label = []
        for i in range(0, len(data)):
            t_id = data[i]
            t_label = label[i]

            tmpd, tmp_l, code = load_track(t_id, t_label)
            if code == 0:
                new_data.append(tmpd)
                new_label.append(tmp_l)

        return new_data, new_label

    def get_data_num(self):
        return self._num_examples

    def get_test_data_num(self):
        return self._test_num

    def next_test_batch(self, batch_size):
        data = self._test_data[self._test_index: min(self._test_index + batch_size, self._test_num)]
        label = self._test_lable[self._test_index: min(self._test_index + batch_size, self._test_num)]
        self._test_index += batch_size
        if self._test_index >= self._test_num:
            self._test_index = 0

        new_data = []
        new_label = []
        for i in range(0, len(data)):
            _id = data[i]
            _label = label[i]
            _tmpd,_tmp_l, code = load_track(_id, _label)
            if code == 0:
                new_data.append(_tmpd)
                new_label.append(_tmp_l)

        return new_data, new_label


class Model:
    def __init__(self):
        self.graph = tf.Graph()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.build_graph()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True

    def _create_placeholders(self):
        with tf.name_scope("input"):
            self.mel = tf.placeholder(tf.float32, shape=[None, input_depth, input_height, input_weight], name='mel')
            self.label = tf.placeholder(tf.float32, shape=[None, label_legth], name='label')
            self.p_c_m = {1: 0, 2: 0, 3: 0}
            self.p_c_v = {1: 0, 2: 0, 3: 0}
            self.p_d_m = {1: 0, 2: 0}
            self.p_d_v = {1: 0, 2: 0}

    def _create_params(self):
        with tf.name_scope("params"):
            with tf.name_scope("cnn"):
                self.learning_rate = learning_rate
                self.C = dict()  # cnn kernel
                self.P = {  # pooling size
                    1: [1, 1, 4, 4, 1],
                    2: [1, 1, 4, 4, 1],
                    3: [1, 1, 2, 2, 1],
                }
                self.c_m = dict()  # cnn mean for batch normalization
                self.c_v = dict()  # cnn variance for batch normalization
                self.c_beta = dict()  # cnn beta for batch normalization
                self.c_gama = dict()  # cnn gama for batch normalization
                self.C[1] = tf.get_variable('C1', [1, 5, 5, 1, 4],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate),
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.c_gama[1] = tf.get_variable('c_gama1', [4],
                                                 initializer=tf.constant_initializer(1.0))
                self.c_beta[1] = tf.get_variable('c_beta1', [4],
                                                 initializer=tf.constant_initializer(0.0))
                self.C[2] = tf.get_variable('C2', [1, 5, 5, 4, 8],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate),
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.c_gama[2] = tf.get_variable('c_gama2', [8],
                                                 initializer=tf.constant_initializer(1.0))
                self.c_beta[2] = tf.get_variable('c_beta2', [8],
                                                 initializer=tf.constant_initializer(0.0))
                self.C[3] = tf.get_variable('C3', [1, 5, 5, 8, 16],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate),
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.c_gama[3] = tf.get_variable('c_gama3', [16],
                                                 initializer=tf.constant_initializer(1.0))
                self.c_beta[3] = tf.get_variable('c_beta3', [16],
                                                 initializer=tf.constant_initializer(0.0))
            with tf.name_scope("dnn"):
                self.W = dict()  # W for dnn
                self.d_m = dict()  # dnn mean for batch normalization
                self.d_v = dict()  # dnn variance for batch normalization
                self.d_beta = dict()  # dnn beta for batch normalization
                self.d_gama = dict()  # dnn gama for batch normalization

                self.W[1] = tf.get_variable('W1', [256, 64],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate),
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.d_gama[1] = tf.get_variable('d_gama1', [64],
                                                 initializer=tf.constant_initializer(1.0))
                self.d_beta[1] = tf.get_variable('d_beta1', [64],
                                                 initializer=tf.constant_initializer(0.0))
                self.W[2] = tf.get_variable('W2', [64, 8],
                                            regularizer=tf.contrib.layers.l2_regularizer(scale=reg_rate),
                                            initializer=tf.contrib.layers.xavier_initializer(seed=1))
                self.d_gama[2] = tf.get_variable('d_gama2', [8],
                                                 initializer=tf.constant_initializer(1.0))
                self.d_beta[2] = tf.get_variable('d_beta2', [8],
                                                 initializer=tf.constant_initializer(0.0))

    @staticmethod
    def batch_normalization(x, offset, scale):
        mean = tf.reduce_mean(x)
        variance = tf.reduce_mean(tf.square(mean - x))
        n_x = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=offset, scale=scale,
                                        variance_epsilon=1e-12)
        return mean, variance, n_x

    def calc_cnn_layer(self, input_layer, level, mean=None, variance=None):
        Z = tf.nn.conv3d(input_layer, self.C[level], strides=[1, 1, 1, 1, 1], padding='SAME')
        if mean is None or variance is None:
            self.c_m[level], self.c_v[level], n_Z = self.batch_normalization(Z, offset=self.c_beta[level],
                                                                             scale=self.c_gama[level])
        else:
            n_Z = tf.nn.batch_normalization(Z, mean=mean, variance=variance, offset=self.c_beta[level],
                                            scale=self.c_gama[level], variance_epsilon=1e-12)
        A = tf.nn.relu(n_Z)
        P = tf.nn.max_pool3d(A, ksize=self.P[level], strides=self.P[level], padding='SAME')
        return P

    def calc_dnn_layer(self, input_layer, level, relu=False, mean=None, variance=None):
        Z = tf.matmul(input_layer, self.W[level])
        if mean is None or variance is None:
            self.d_m[level], self.d_v[level], ret = self.batch_normalization(Z, offset=self.d_beta[level],
                                                                             scale=self.d_gama[level])
        else:
            ret = tf.nn.batch_normalization(Z, mean=mean, variance=variance, offset=self.d_beta[level],
                                            scale=self.d_gama[level], variance_epsilon=1e-12)
        if relu:
            ret = tf.nn.relu(ret)
        return ret

    def get_output_layer(self, input_layer, c_m=None, c_v=None, d_m=None, d_v=None):
        cnn_layer = dict()
        dnn_layer = dict()
        cnn_layer[0] = tf.reshape(input_layer, [-1, input_depth, input_height, input_weight, 1])
        for level in range(1, 4):
            if c_m is not None and c_v is not None:
                cnn_layer[level] = self.calc_cnn_layer(cnn_layer[level - 1], level, c_m[level], c_v[level])
            else:
                cnn_layer[level] = self.calc_cnn_layer(cnn_layer[level - 1], level)
        dnn_layer[0] = tf.contrib.layers.flatten(
            tf.nn.avg_pool3d(cnn_layer[3], ksize=[1, input_depth, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='VALID'))
        if d_m is not None and d_v is not None:
            dnn_layer[1] = self.calc_dnn_layer(dnn_layer[0], 1, True, d_m[1], d_v[1])
            dnn_layer[2] = self.calc_dnn_layer(dnn_layer[1], 2, False, d_m[2], d_v[2])
        else:
            dnn_layer[1] = self.calc_dnn_layer(dnn_layer[0], 1, True)
            dnn_layer[2] = self.calc_dnn_layer(dnn_layer[1], 2)
        return dnn_layer[2]

    def _create_loss(self):
        with tf.name_scope("loss"):
            y = self.get_output_layer(self.mel)
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=self.label)) + self.reg_loss

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def _predict(self):
        with tf.name_scope("predict"):
            #y = self.get_output_layer(self.mel, c_m=self.p_c_m, c_v=self.p_c_v, d_m=self.p_d_m, d_v=self.p_d_v)
            y = self.get_output_layer(self.mel)
            self.p_y = y
            #self.num = tf.reduce_sum(1 - tf.to_int32((tf.argmax(y, -1) - tf.argmax(self.label, -1)) > 0))
            self.total = tf.reduce_sum(tf.to_int32(tf.argmax(self.label, 1) >= 0))
            self.num = tf.reduce_sum(tf.to_int32(1 - tf.abs(tf.to_int32(tf.argmax(y, 1) - tf.argmax(self.label, 1))) > 0))
            #self.total = tf.reduce_sum(tf.to_int32(tf.argmax(self.label, -1)) >= 0)


    def build_graph(self):
        with self.graph.as_default() as g:
            with g.device('/cpu:0'):
                self._create_placeholders()
                self._create_params()
            with g.device('/gpu:0'):
                self._create_loss()
                self._create_optimizer()
                self._predict()
            with g.device('/cpu:0'):
                self._create_summaries()

    @staticmethod
    def update_avg(x, _x, t):
        ret = {}
        for i in x:
            ret[i] = (x[i] * moving_avg_rate + (1 - moving_avg_rate) * _x[i]) / (1 - math.pow(moving_avg_rate, t))
        return ret

    def train(self, train_data, batch_size=8):
        model_dir = './model/'
        os.system('rm -rf ' + model_dir)
        with tf.Session(graph=self.graph, config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(model_dir + 'board', sess.graph)
            step = 0
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            for epoch in range(EPOCHES):
                cm = {1: 0, 2: 0, 3: 0}
                cv = {1: 0, 2: 0, 3: 0}
                dm = {1: 0, 2: 0}
                dv = {1: 0, 2: 0}
                total_loss = 0.0
                total_num = 0
                total_total = 0
                for batch in tqdm(range(train_data.get_data_num() / batch_size + 1)):
                    mel, label = train_data.next_batch(batch_size)
                    feed_dict = {
                        self.mel: mel,
                        self.label: label
                    }
                    loss, _, summary, _cm, _cv, _dm, _dv, _num, _total = sess.run([self.loss, self.optimizer, self.summary_op, self.c_m, self.c_v, self.d_m, self.d_v, self.num, self.total], feed_dict=feed_dict)
                    cm = self.update_avg(cm, _cm, batch + 1)
                    cv = self.update_avg(cv, _cv, batch + 1)
                    dm = self.update_avg(dm, _dm, batch + 1)
                    dv = self.update_avg(dv, _dv, batch + 1)

                    total_loss += loss
                    total_num += _num
                    total_total += _total
                    step += 1

                    if step % 10 == 0:

                        writer.add_summary(summary, global_step=step)
                        writer.flush()

                #print cm, cv, dm, dv
                print total_loss, total_num, total_total, total_num * 1.0 / total_total
                saver.save(sess, model_dir + 'model', global_step=step)

                num = 0
                total = 0
                for _ in tqdm(range((train_data.get_test_data_num() - 1) / batch_size + 1)):
                    mel, label = train_data.next_test_batch(batch_size)
                    self.p_c_m = cm
                    self.p_c_v = cv
                    self.p_d_m = dm
                    self.p_d_v = dv
                    feed_dict = {
                        self.mel: mel,
                        self.label: label,
                    }
                    _num, _total, _y = sess.run([self.num, self.total, self.p_y], feed_dict=feed_dict)

                    yy = np.argmax(_y, 1)
                    ll = np.argmax(label, 1)

                    _num = (yy == ll).sum()
                    num += _num
                    total += _total

                print num, total, num * 1.0 / total


if __name__=="__main__":
    # train_data = TrainData()
    # train_data.load_data("/data/raw-data/audio_feature/mel-16-128-18/train_data_16_128_128.npy",
    #                      "/data/raw-data/audio_feature/mel-16-128-18/train_label_16_128_128.npy")
    data = np.load(sys.argv[1])
    data = data.tolist()

    indexs = [i for i in range(0, len(data["y"]))]

    np.random.shuffle(indexs)

    shuf_track_paths = []
    shuf_track_y = []
    track_paths = data['track_paths']
    y = data["y"]
    for i in range(0, len(indexs)):
        shuf_track_paths.append(track_paths[indexs[i]])
        shuf_track_y.append(y[indexs[i]])

    print("here", shuf_track_paths[0], shuf_track_y[0])

    train_data = TrainBigData()
    train_data.load_data(train_data=shuf_track_paths, train_label=shuf_track_y)
    model = Model()
    model.train(train_data)
