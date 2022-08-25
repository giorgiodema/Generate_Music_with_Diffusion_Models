from params import params
import matplotlib.pyplot as plt
import numpy as np

params["MODEL_NAME"] = "diff_wave_net"
TRAIN_SIZE = 1000#1248#1498
PRINT_EVERY = 1000

with open(f"log/{params['MODEL_NAME']}.txt", "r") as f:
    l = f.read().split(",")
l = list(filter(lambda x:x!="",l))
l = list(map(lambda x:float(x),l))
plt.plot(l)
plt.title(f'{params["MODEL_NAME"]} train loss')
plt.show()


a = np.repeat(np.array(l),PRINT_EVERY)
size = (a.shape[0]//TRAIN_SIZE) * TRAIN_SIZE
a = a[0:size]
a = np.reshape(a,(a.shape[0]//TRAIN_SIZE,TRAIN_SIZE))
a = np.mean(a,axis=1,keepdims=False)
plt.plot(a)
plt.title(f'{params["MODEL_NAME"]} train loss avg {TRAIN_SIZE}')
plt.show()