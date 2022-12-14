from diffusion.diffusion_process import *
from network.model import ConditionedDiffWaveNet
import tensorflow as tf
import os
from params import params
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(2)

# NB: SELECT THE TRAIN STEP FROM WHICH YOU WANT TO
#     RESTORE THE MODEL
training_step_ckpt=1258320
##################################################
                       
net =  ConditionedDiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])
params["MODEL_NAME"] = net.name

NSAMPLES = 660000
beta = variance_schedule(params["DIFF_STEPS"])
alpha = get_alpha(beta)
alpha_hat = get_alpha_hat(alpha)
beta_hat = get_beta_hat(alpha_hat,beta)

print("-------------------------------------------")
print("-------------------------------------------")
print(f"Model:{net.name}")
ret = net.load_weights(f"ckpt/__step_{training_step_ckpt}__{params}")

while True:
    print("-> Listening Generated Song")
    genre = input("Genre:")
    x_0_gen = conditioned_backward_process(net,(1,NSAMPLES,1),genre,params["DIFF_STEPS"])
    wav = get_wav(x_0_gen[0],SR//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())