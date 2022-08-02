from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import *
import tensorflow as tf
import os
import subprocess

params = {
    "BS":4,
    "DIFF_STEPS":20,
    "DEPTH":10,
    "CHANNELS":64,
    "KERNEL_SIZE":3,
    "NSPLITS":6,
    "DOWNSAMPLE":3,
    "SR":22050,
    "NSAMPLES":660000
}

tf.random.set_seed(2)

ds = get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"])
it = iter(ds)
x_0 = next(it)

wav = get_wav(x_0[0],params["SR"]//params["DOWNSAMPLE"])
subprocess.run(["ffplay","-"],input=wav.numpy())

beta = variance_schedule(params["DIFF_STEPS"])
alpha = get_alpha(beta)
alpha_hat = get_alpha_hat(alpha)
beta_hat = get_beta_hat(alpha_hat,beta)

for i in range(params["DIFF_STEPS"]):
    t = i
    print(f"\n\n--------------- DIFF STEP {t:3d} ---------------\n\n")
    inp,_ = forward(x_0,alpha_hat,t)
    wav = get_wav(inp[0],params["SR"]//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())