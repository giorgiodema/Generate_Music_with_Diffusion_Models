from diffusion.diffusion_process import *
from network.model import DiffWaveNet, DnCNN, SimpleRecurrentResNet, SimpleResNet
from data.dataset import get_unlabelled_dataset
import tensorflow as tf
import os
from params import params

tf.get_logger().setLevel('ERROR')

ELEMENT_SHAPE = (params["BS"],(params["SR"]//params["NSPLITS"])//params["DOWNSAMPLE"]+1,1)
net = DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])
params["MODEL_NAME"] = net.name
train(
    get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"]),
    params["DIFF_STEPS"],
    net,
    tf.keras.optimizers.Adam(learning_rate=2*10**-4),
    str(params)
)