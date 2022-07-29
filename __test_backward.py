from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import *
import tensorflow as tf
import os
import subprocess
from params import params

START_STEP = 10
PREFIX="__best__"#"__last__"
ELEMENT_SHAPE = (params["BS"],(params["SR"]//params["NSPLITS"])//params["DOWNSAMPLE"]+1,1)

ds = get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"])
it = iter(ds)
x_0 = next(it)

net = tf.keras.models.load_model(f"ckpt/{PREFIX}{str(params)}")
net.summary()

print("----------------- ORIGINAL -----------------")
wav = get_wav(x_0[0],SR//params["DOWNSAMPLE"])
subprocess.run(["ffplay","-"],input=wav.numpy())

beta = variance_schedule(params["DIFF_STEPS"])
alpha = get_alpha(beta)
alpha_hat = get_alpha_hat(alpha)
beta_hat = get_beta_hat(alpha_hat,beta)

x_0_forw = forward(x_0,alpha_hat,START_STEP)
print("----------------- FORWARD -----------------")
wav = get_wav(x_0_forw[0],SR//params["DOWNSAMPLE"])
subprocess.run(["ffplay","-"],input=wav.numpy())

x_0_back = backward_process_from(net,ELEMENT_SHAPE,params["DIFF_STEPS"],x_0_forw,START_STEP)
print("----------------- BACKWARD -----------------")
wav = get_wav(x_0_back[0],SR//params["DOWNSAMPLE"])
subprocess.run(["ffplay","-"],input=wav.numpy())
