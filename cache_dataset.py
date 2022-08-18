from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import get_unlabelled_dataset
import tensorflow as tf
import os
from params import params


ds = get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"]).take(10)
tf.data.experimental.save(
ds, "./dataset/unlabelled")