from params import params
import matplotlib.pyplot as plt

params["MODEL_NAME"] = "diff_wave_net"

with open(f"log/{params}.txt", "r") as f:
    l = f.read().split(",")
l = list(filter(lambda x:x!="",l))
l = list(map(lambda x:float(x),l))
plt.plot(l)
plt.title(f'{params["MODEL_NAME"]} train loss')
plt.show()

