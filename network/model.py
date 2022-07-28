from unittest import skip
import tensorflow as tf

class ResidualLayer(tf.keras.Model):
    def __init__(self, dilation, C, kernel_size):
        super().__init__()
        self.dil_conv = tf.keras.layers.Conv1D(C,kernel_size,padding='same',dilation_rate=dilation)
        self.fc_emb = tf.keras.layers.Dense(C,activation=tf.keras.activations.swish)
        self.res_conv = tf.keras.layers.Conv1D(C,1,padding='same')
        self.skip_conv = tf.keras.layers.Conv1D(C,1,padding='same')

    def call(self,inputs):
        """
        Residual Block,
        input.shape -> (B,1,L)
        emb.shape   -> (B,512)
        """                                             # Shapes
        input = inputs[0]
        step_embedding = inputs[1]
        emb = self.fc_emb(step_embedding)               # (B,C)
        emb = tf.expand_dims(emb,1)                    # (B,1,C)
        emb = tf.repeat(
            emb,
            repeats=tf.shape(input)[1],
            axis=1)                                     # (B,L,C)
        inp_emb = input + emb                           # (B,L,C)
        conv = self.dil_conv(inp_emb)                   # (B,L,C)
        #if conditioner_embedding:
        #    conv = conv + conditioner_embedding
        tanh = tf.keras.activations.tanh(conv)
        sigm = tf.keras.activations.sigmoid(conv)
        o = tanh * sigm
        res_out = self.res_conv(o)
        res_out = input + res_out                             # (B,L,C)
        skip_out = self.skip_conv(o)                                    # (B,L,C)
        return res_out,skip_out


class DiffWaveNet(tf.keras.Model):
    def __init__(self, depth, C, kernel_size):
        super(DiffWaveNet,self).__init__()
        self.depth = depth
        self.input_conv = tf.keras.layers.Conv1D(C,1,padding='same',activation='relu')
        self.dens_emb_1 = tf.keras.layers.Dense(512,activation=tf.keras.activations.swish)
        self.dens_emb_2 = tf.keras.layers.Dense(512,activation=tf.keras.activations.swish)
        self.res_blocks = []
        for i in range(depth):
            self.res_blocks.append(
                ResidualLayer(2**i,C,kernel_size)
            )
        self.out_conv1 = tf.keras.layers.Conv1D(C,kernel_size,padding='same',activation='relu')
        self.out_conv2 = tf.keras.layers.Conv1D(1,kernel_size,padding='same')

    def call(self, inputs):
        input = inputs[0]
        step_embedding = inputs[1]
        input = self.input_conv(input)                             # (B,L,C)
        emb = tf.expand_dims(step_embedding,0)                     # (1,128)
        emb = tf.repeat(emb,repeats=tf.shape(input)[0],axis=0)     # (B,128)
        emb = self.dens_emb_1(emb)
        emb = self.dens_emb_2(emb)
        res_input = input
        skip_outputs = tf.TensorArray(
            tf.float32,
            size=self.depth,
            clear_after_read=True)
        for i in range(len(self.res_blocks)):
            res_input,skip_out = self.res_blocks[i]([res_input,emb])
            skip_outputs = skip_outputs.write(i,skip_out)
        out = skip_outputs.stack()                      # (depth,B,L,C)
        out = tf.reduce_sum(out,axis=0)                 # (B,L,C)
        out = self.out_conv1(out)                       # (B,L,C)
        out = self.out_conv2(out)                       # (B,L,1)
        return out
