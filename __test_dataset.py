import tensorflow as tf
import math
from params import params
from data.dataset import get_wav
import subprocess

ds = tf.data.experimental.load("./dataset/unlabelled_10",tf.TensorSpec(shape=(params["BS"],math.ceil((params["NSAMPLES"]//params["NSPLITS"])/params["DOWNSAMPLE"]),1), dtype=tf.float32))
for x in ds:
    for i in range(x.shape[0]):
        
        wav = get_wav(x[i],params["SR"]//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())