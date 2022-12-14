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

ELEMENT_SHAPE = (params["BS"],(params["NSAMPLES"]//params["NSPLITS"])//params["DOWNSAMPLE"]+1,1)
nets =[ DiffWaveNet(params["DEPTH"],params["CHANNELS"],params["KERNEL_SIZE"]),
        SimpleResNet((ELEMENT_SHAPE[1],1)),
        DnCNN((ELEMENT_SHAPE[1],1))]

for net in nets:
    tf.keras.backend.clear_session()
    if not os.path.exists(f"test/{net.name}.txt"):
        train_single_sample(
            x_0,
            params["DIFF_STEPS"],
            net,
            tf.keras.optimizers.Adam(learning_rate=2*10**-4),
            str(net.name),
            training_steps=10**5
        )
        net.save_weights(f"test/model_test/{net.name}")

START_STEP = 5#params["DIFF_STEPS"]-1
beta = variance_schedule(params["DIFF_STEPS"])
alpha = get_alpha(beta)
alpha_hat = get_alpha_hat(alpha)
beta_hat = get_beta_hat(alpha_hat,beta)
for net in nets:
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
    
    x_0_forw,_ = forward(x_0,alpha_hat,START_STEP)
    print("-> Listening Noisy song")
    for i in range(x_0_forw.shape[0]):
        wav = get_wav(x_0_forw[i],params["SR"]//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())

    x_0_back = backward_process_from(net,ELEMENT_SHAPE,params["DIFF_STEPS"],x_0_forw,START_STEP)
    print("-> Listening Denoised")
    for i in range(x_0_back.shape[0]):
        wav = get_wav(x_0_back[i],params["SR"]//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())

    print("-> Listening Generated Song")
    x_0_gen = backward_process(net,ELEMENT_SHAPE,params["DIFF_STEPS"])
    for i in range(x_0_gen.shape[0]):
        wav = get_wav(x_0_gen[i],params["SR"]//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())