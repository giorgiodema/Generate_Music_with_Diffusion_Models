from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import *
import tensorflow as tf
import os
import subprocess
from params import params
import math

START_STEP = 5
MODEL_STEP = 269568
ELEMENT_SHAPE = (params["BS"],math.ceil((params["NSAMPLES"]//params["NSPLITS"])/params["DOWNSAMPLE"]),1)

ds = tf.data.experimental.load("./dataset/unlabelled",tf.TensorSpec(shape=(params["BS"],math.ceil((params["NSAMPLES"]//params["NSPLITS"])/params["DOWNSAMPLE"]),1), dtype=tf.float32))
it = iter(ds)

net =  DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])
params["MODEL_NAME"] = net.name
net.load_weights(f"ckpt/__step_{MODEL_STEP}__{params}")
while True:
    x_0 = next(it)
    print("----------------- ORIGINAL -----------------")
    wav = get_wav(x_0[0],params["SR"]//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())

    beta = variance_schedule(params["DIFF_STEPS"])
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    beta_hat = get_beta_hat(alpha_hat,beta)

    x_0_forw,_ = forward(x_0,alpha_hat,START_STEP)
    print("----------------- FORWARD -----------------")
    wav = get_wav(x_0_forw[0],params["SR"]//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())

    x_0_back = backward_process_from(net,ELEMENT_SHAPE,params["DIFF_STEPS"],x_0_forw,START_STEP)
    print("----------------- BACKWARD -----------------")
    wav = get_wav(x_0_back[0],params["SR"]//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())
