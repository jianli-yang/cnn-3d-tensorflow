# -*- coding: UTF-8 -*-
import numpy as np
import os, json


def load_data(file_name, y):
    #file_name = os.path.join(os.path.abspath(os.curdir), file_name)
    data = np.load(file_name)
    dec = 128
    base = 500
    my_list = []
    for i in range(16):
        # r_data = data[:, base:base + dec]
        r_data = data[base:base + dec, :]
        base += 64

        # print r_data.shape
        r_data = r_data.reshape(1, -1)[0].tolist()
        if len(r_data) != 128 * 128:
            return None, None, -1
        my_list.append(r_data)
    if len(y) == 1:
        _y = [0, 0, 0]
        _y[int(y)] = 1
    else:
        _y = y

    code = 0
    if len(my_list) != 16:
        code = -2
    return my_list, _y, code


def read_file(file_name):
    message = []
    with open(file_name) as f:
        while True:
            line = f.readline().strip()
            if not line:
                break
            params = line.split(',')
            message.append(params)
    print(len(message))

    all_data = []
    all_label = []

    ids = []
    labels = []
    np.random.shuffle(message)
    cnt = 0
    for msg in message:
        print "process {} {} ...".format(cnt, msg)
        cnt += 1
        d_line = "/data/raw-data/audio_feature/featrues-300/{}.m4a.f.npy".format(msg[0])
        data, label = load_data(d_line, msg[1])
        all_data.append(data)
        all_label.append(label)

        ids.append(int(msg[0]))
        labels.append(int(msg[1]))

    print "write to file ..."
    np.save("/data/raw-data/audio_feature/mel-16-128-18/train_data_16_128_128", np.array(all_data).reshape(-1, 16, 128, 128))
    np.save("/data/raw-data/audio_feature/mel-16-128-18/train_label_16_128_128", np.array(all_label))
    np.save("/data/raw-data/audio_feature/feature-300-id", np.array(ids))
    np.save("/data/raw-data/audio_feature/feature-300-label", np.array(labels))
    print "write to file ok"


def load_track(id, label):
    d_line = "{}.mel.npy".format(id)
    data, label, code = load_data(d_line, label)
    if code == 0:
        data = np.array(data).reshape(16, 128, 128)
        data = data * 1.0 / data.max()
        label = np.array(label)
    return data, label, code

# if __name__ == "__main__":
#     read_file(os.sys.argv[1])


