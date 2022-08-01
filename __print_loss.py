from params import params
import matplotlib.pyplot as plt
import numpy as np

params["MODEL_NAME"] = "conditioned_diff_wave_net"

with open(f"log/{params}.txt", "r") as f:
    l = f.read().split(",")
l = list(filter(lambda x:x!="",l))
l = list(map(lambda x:float(x),l))
plt.plot(l)
plt.title(f'{params["MODEL_NAME"]} train loss')
plt.show()

resolution = 1498
a = np.array(l)
size = (a.shape[0]//resolution) * resolution
a = a[0:size]
a = np.reshape(a,(a.shape[0]//resolution,resolution))
a = np.mean(a,axis=1,keepdims=False)
plt.plot(a)
plt.title(f'{params["MODEL_NAME"]} train loss avg {resolution}')
plt.show()