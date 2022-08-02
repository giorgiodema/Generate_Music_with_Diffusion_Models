from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import get_unlabelled_dataset
import tensorflow as tf
import os
import math
from params import params

tf.get_logger().setLevel('ERROR')

net = DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])
params["MODEL_NAME"] = net.name
ds = tf.data.experimental.load("./dataset/unlabelled",tf.TensorSpec(shape=(params["BS"],math.ceil((params["NSAMPLES"]//params["NSPLITS"])/params["DOWNSAMPLE"]),1), dtype=tf.float32))
train(
    ds,
    params["DIFF_STEPS"],
    net,
    tf.keras.optimizers.Adam(learning_rate=2*10**-4),
    str(params)
)