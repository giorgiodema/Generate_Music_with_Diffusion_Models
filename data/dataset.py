import os
import tensorflow as tf
import numpy as np

NSAMPLES=660000
SR = 22050

labels = tf.constant(["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"])
indices = tf.constant([0,1,2,3,4,5,6,7,8,9])



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
    waveform,_ = tf.audio.decode_wav(binary,desired_samples=NSAMPLES)
    return waveform

def low_pass(signal,fc,npoints=128):
    fc = tf.cast(fc,tf.float32)
    omegac = 2*np.pi*fc/SR
    nrange = tf.range(0,npoints,dtype=tf.float32) - npoints//2
    hn = tf.sin(omegac * nrange) / (np.pi * nrange)
    wn = 0.54 + 0.46 * tf.cos(2*np.pi*nrange/npoints)
    kernel = hn * wn
    indices = tf.where(tf.math.is_nan(kernel))
    kernel = tf.tensor_scatter_nd_update(
        kernel,
        indices,
        tf.ones((tf.shape(indices)[0]))*0.5
    )
    kernel = tf.expand_dims(kernel,-1)
    kernel = tf.expand_dims(kernel,-1)
    signal = tf.expand_dims(signal,0)
    sfiltered = tf.nn.conv1d(signal,kernel,padding='SAME',stride=1)
    return sfiltered[0]

def get_waveform_split(file_path,nsplits,splitid,downsample):
    binary = tf.io.read_file(file_path)
    waveform,_ = tf.audio.decode_wav(binary,desired_samples=NSAMPLES)
    samples_per_split = NSAMPLES//nsplits
    waveform_split = waveform[splitid*samples_per_split:(splitid+1)*samples_per_split,:]
    if downsample>1:
        waveform_split = low_pass(waveform_split,SR//(downsample*2))
        waveform_split = waveform_split[::downsample,:]
    return waveform_split

def get_waveform_and_sr(file_path):
    binary = tf.io.read_file(file_path)
    waveform,sr = tf.audio.decode_wav(binary,desired_samples=NSAMPLES)
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
"""
def get_unlabelled_dataset(bs=None):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(get_waveform)
    if bs:
        ds = ds.batch(bs,drop_remainder=True)
    return ds
"""
def get_unlabelled_dataset(bs=None,nsplits=1,downsample=1):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    if nsplits==1:
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda x:get_waveform_split(x,1,0,downsample))
    elif nsplits > 1:
        datasets = []
        samples_per_split = NSAMPLES//nsplits
        for splitid in range(nsplits):
            if samples_per_split*splitid + samples_per_split < NSAMPLES:
                ds = tf.data.Dataset.from_tensor_slices(filenames)
                ds = ds.map(lambda x: get_waveform_split(x,nsplits,splitid,downsample))
                datasets.append(ds)
        ds = datasets[0]
        for dsp in datasets[1:]:
            ds = ds.concatenate(dsp)
    else:
        raise ValueError("Invalid argument nsplit")
    
    if bs:
        ds = ds.batch(bs,drop_remainder=True)
    return ds

"""
def get_labelled_dataset(bs=None):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(lambda x:(get_waveform(x),get_label(x)))
    if bs:
        ds = ds.batch(bs,drop_remainder=True)
    return ds
"""

def get_labelled_dataset(bs=None,nsplits=1,downsample=1):
    filenames = get_filenames()
    filenames = tf.random.shuffle(filenames)
    if nsplits==1:
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        ds = ds.map(lambda x: (get_waveform_split(x,1,0,downsample),get_label(x)))
    elif nsplits > 1:
        datasets = []
        for splitid in range(nsplits):
            ds = tf.data.Dataset.from_tensor_slices(filenames)
            ds = ds.map(lambda x: (get_waveform_split(x,nsplits,splitid,downsample=downsample),get_label(x)))
            datasets.append(ds)
        ds = datasets[0]
        for dsp in datasets[1:]:
            ds = ds.concatenate(dsp)
    else:
        raise ValueError("Invalid argument nsplit")
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(labels,indices),
        -1
    )
    ds = ds.map(lambda x,y:(x,table.lookup(y)))
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
    downsample = 3
    ds = get_labelled_dataset(4,nsplits=12,downsample=downsample)
    for x,l in ds:
        for i in range(x.shape[0]):
            
            wav = get_wav(x[i],SR//downsample)
            print(f"-----------------\n{labels[l[i]]}\n-----------------\n")
            subprocess.run(["ffplay","-"],input=wav.numpy())

            """
            wav = get_wav(x[i],SR)
            print(f"-----------------\n{l[i]}\n-----------------\n")
            subprocess.run(["ffplay","-"],input=wav.numpy())
            wav = get_wav(low_pass(x[i],SR//2),SR)
            subprocess.run(["ffplay","-"],input=wav.numpy())

            wav = low_pass(x[i],SR//2)
            wav = wav[::2,:]
            wav = get_wav(wav,SR//2)
            subprocess.run(["ffplay","-"],input=wav.numpy())
            """