import tensorflow as tf

def encode(pos,dmodel,n=10000):
    """
    The function returns the positional encoding of position pos, as implemented in 
    https://arxiv.org/pdf/1706.03762.pdf. The dimension of the encoding is dmodel.
    """
    pos = tf.cast(pos,tf.float32)
    dmodel = tf.cast(dmodel,tf.float32)
    i = tf.range(start=0,limit=dmodel//2)
    den = (2*i)/dmodel
    even_val = tf.math.sin(pos/(n**den))
    odd_val  = tf.math.cos(pos/(n**den))

    even_idx = tf.expand_dims(tf.range(start=0,limit=dmodel,delta=2,dtype=tf.int32),-1)
    odd_idx  = tf.expand_dims(tf.range(start=1,limit=dmodel,delta=2,dtype=tf.int32),-1)

    encoding = tf.scatter_nd(indices=even_idx,updates=even_val,shape=(dmodel,))
    encoding += tf.scatter_nd(indices=odd_idx,updates=odd_val,shape=(dmodel,))
    return encoding


if __name__=="__main__":
    import matplotlib.pyplot as plt
    dmodel=128
    encodings = []
    for k in range(100):
        # convert to tensor to avoid tf.function retracing
        encodings += [encode(tf.convert_to_tensor(k,tf.float32),tf.convert_to_tensor(dmodel,tf.float32))]
    encodings = tf.stack(encodings,axis=0)
    cax = plt.matshow(encodings)
    plt.gcf().colorbar(cax)
    plt.savefig("tmp/encodings.png")
    plt.clf()
