import os
import tensorflow as tf
import numpy as np

def get_filenames():
    filenames = []
    base_path = "dataset/Data/genres_original"
    for d in os.listdir(base_path):
        p = os.path.join(base_path,d)
        for filename in os.listdir(p):
            filenames.append(os.path.join(p,filename))
    return filenames

def get_waveform(file_path):
    binary = tf.io.read_file(file_path)
    waveform,_ = tf.audio.decode_wav(binary,desired_samples=660000)
    return waveform

def get_waveform_and_sr(file_path):
    binary = tf.io.read_file(file_path)
    waveform,sr = tf.audio.decode_wav(binary,desired_samples=660000)
    return waveform,sr

def get_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    l = tf.strings.split(
        input=parts[-1],
        sep="."
    )[0]

    return l

def save_wav(waveform,sr,filename):
    encoded = tf.audio.encode_wav(waveform,sr)
    tf.io.write_file(filename,encoded)

def get_wav(waveform,sr):
    encoded = tf.audio.encode_wav(waveform,sr)
    return encoded

def get_unlabelled_dataset(bs=None):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(get_waveform)
    if bs:
        ds.batch(bs,drop_remainder=True)
    return ds

def get_labelled_dataset(bs=None):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(lambda x:(get_waveform(x),get_label(x)))
    if bs:
        ds = ds.batch(bs,drop_remainder=True)
    return ds

def get_min_max_lengths(fnames):
    minl = np.inf
    maxl = 0
    for f in fnames:
        wav = get_waveform(f)
        l = wav.shape[0]
        minl = min(minl,l)
        maxl = max(maxl,l)
    return minl,maxl


if __name__=="__main__":

    import subprocess
    fnames = get_filenames()
    _,sr = get_waveform_and_sr(fnames[0])
    ds = get_labelled_dataset(4)
    for x,l in ds:
        for i in range(x.shape[0]):
            wav = get_wav(x[i],sr)
            print(f"-----------------\n{l[i]}\n-----------------\n")
            subprocess.run(["ffplay","-"],input=wav.numpy())
