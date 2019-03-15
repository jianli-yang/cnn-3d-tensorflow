# -*- coding: UTF-8 -*-
import librosa as lbr
import beanstalkc
import os, sys, json, time
import numpy as np

path = "/data/jianli.yang/music-raw-data-8000/wavs/"

## 提取音频的mel频谱
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

beanstalk = beanstalkc.Connection("box.beanstalk.starmaker.co", 11300)
tube = "get_media_mel"


def read_file(filename):
    res = []
    with open(filename) as f:
        while True:
            lines = f.readlines(1000)
            if not lines:
                break
            beanstalk.use(tube)
            for line in lines:
                line = line.strip()
                data = line.split('\t')
                beanstalk.put(data[1])
    return res


def write2file(filename, data):
    print("write2file {}".format(filename))
    with open(filename, "w") as f:
        f.write(data)


def get_mel(filename):
    start = time.clock()
    filename_all = os.path.join(path, filename + ".wav")
    print("now process {}".format(filename_all))
    new_input, sample_rate = lbr.load(filename_all, mono=True)
    features = lbr.feature.melspectrogram(new_input, **MEL_KWARGS).T

    data_file = os.path.join(path, filename + ".mel")
    print("now save {}".format(data_file))
    np.save(data_file, features)
    print("time_eps={}".format(time.clock() - start))
    # features = json.dumps(features)
    #write2file(data_file, features)


def get_multi_mel():
    while True:
        beanstalk.watch(tube)
        payload = beanstalk.reserve(5)
        if not payload:
            sys.exit(0)
        print("------------------------------\n")
        msg = payload.body
        msg = str(msg).strip()
        get_mel(msg)
        payload.delete()
        print("------------------------------\n\n")


if __name__ == "__main__":
    get_multi_mel()

