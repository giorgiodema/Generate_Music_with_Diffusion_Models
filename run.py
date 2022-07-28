from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import get_unlabelled_dataset
import tensorflow as tf
import os

tf.get_logger().setLevel('ERROR')
params = {
    "BS":8,
    "DIFF_STEPS":1000,
    "DEPTH":6,
    "CHANNELS":64,
    "KERNEL_SIZE":3,
    "NSPLITS":6,
    "DOWNSAMPLE":3
}

net = DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])
train(
    get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"]),
    params["DIFF_STEPS"],
    net,
    tf.keras.optimizers.Adam(),
    str(params),
    epochs=1000
)