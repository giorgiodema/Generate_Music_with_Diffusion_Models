from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import get_unlabelled_dataset,SR,get_wav
import tensorflow as tf
import subprocess

tf.get_logger().setLevel('ERROR')
params = {
    "BS":4,
    "DIFF_STEPS":1000,
    "DEPTH":6,
    "CHANNELS":64,
    "KERNEL_SIZE":3,
    "NSPLITS":3,
    "DOWNSAMPLE":3
}

ELEMENT_SHAPE = (4, 73334, 1)
RET_SEQ=False

net = tf.keras.models.load_model(f"ckpt/{str(params)}")
net.summary()
while True:
    if RET_SEQ:
        seq = sample(
            net,
            ELEMENT_SHAPE,
            params["DIFF_STEPS"],
            return_sequence=True
        )
        for i,x in enumerate(seq[-10:]):
            print(f"----------\nDIFFUSION STEP {i:2d}----------\n")
            for i in range(x.shape[0]):
                wav = get_wav(x[i],SR//params["DOWNSAMPLE"])
                subprocess.run(["ffplay","-"],input=wav.numpy())

        print("\n\n")
    else:
        x = sample(
            net,
            ELEMENT_SHAPE,
            params["DIFF_STEPS"],
            return_sequence=False
        )
        wav = get_wav(x[0],SR//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())