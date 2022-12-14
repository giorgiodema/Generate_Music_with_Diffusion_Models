from diffusion.diffusion_process import *
from network.model import DiffWaveNet, DnCNN, SimpleRecurrentResNet, SimpleResNet
from data.dataset import get_unlabelled_dataset
import tensorflow as tf
import os
import matplotlib.pyplot as plt

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

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(2)

ds = get_unlabelled_dataset(params["BS"],nsplits=params["NSPLITS"],downsample=params["DOWNSAMPLE"])
it = iter(ds)
for i,x_0 in enumerate(it):
    x_0 = next(it)
    if i==7:
        break
#for i in range(params["BS"]):
#    wav = get_wav(x_0[i],SR//params["DOWNSAMPLE"])
#    subprocess.run(["ffplay","-"],input=wav.numpy())

net =  DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"])

NSAMPLES = 660000
beta = variance_schedule(params["DIFF_STEPS"])
alpha = get_alpha(beta)
alpha_hat = get_alpha_hat(alpha)
beta_hat = get_beta_hat(alpha_hat,beta)

print("-------------------------------------------")
print("-------------------------------------------")
print(f"Model:{net.name}")
with open(f"test/{net.name}.txt","r") as f:
    l = f.read().split(",")
l = list(filter(lambda x:x!="",l))
l = list(map(lambda x:float(x),l))
plt.plot(l)
plt.title(f"{net.name} train loss")
plt.show()

net.load_weights(f"test/model_test/{net.name}")

while True:
    print("-> Listening Generated Song")
    x_0_gen = backward_process(net,(1,NSAMPLES,1),params["DIFF_STEPS"])
    wav = get_wav(x_0_gen[0],params["SR"]//params["DOWNSAMPLE"])
    subprocess.run(["ffplay","-"],input=wav.numpy())