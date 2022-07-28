from diffusion.diffusion_process import *
from network.model import DiffWaveNet
from data.dataset import get_unlabelled_dataset,SR,get_wav
import tensorflow as tf
import subprocess
from params import params

tf.get_logger().setLevel('ERROR')

ELEMENT_SHAPE = (params["BS"],(params["SR"]//params["NSPLITS"])//params["DOWNSAMPLE"]+1,1)
RET_SEQ=False
PREFIX="__best__" #"__last__"

net = tf.keras.models.load_model(f"ckpt/{PREFIX}{str(params)}")
net.summary()
while True:
    if RET_SEQ:
        seq = backward_process(
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
        x = backward_process(
            net,
            ELEMENT_SHAPE,
            params["DIFF_STEPS"],
            return_sequence=False
        )
        wav = get_wav(x[0],SR//params["DOWNSAMPLE"])
        subprocess.run(["ffplay","-"],input=wav.numpy())