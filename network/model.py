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
        emb = tf.expand_dims(emb,1)                     # (B,1,C)
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
        skip_out = self.skip_conv(o)                          # (B,L,C)
        return res_out,skip_out

class ConditionedResidualLayer(tf.keras.Model):
    def __init__(self, dilation, C, kernel_size):
        super().__init__()
        self.dil_conv = tf.keras.layers.Conv1D(C,kernel_size,padding='same',dilation_rate=dilation)
        self.fc_emb = tf.keras.layers.Dense(C,activation=tf.keras.activations.swish)
        self.fc_cond = tf.keras.layers.Dense(C,activation=tf.keras.activations.swish)
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
        conditioner = inputs[2]

        conditioner = self.fc_cond(conditioner)         # (B,C)
        conditioner = tf.expand_dims(conditioner,1)     # (B,1,C)
        conditioner = tf.repeat(
            conditioner,
            repeats=tf.shape(input)[1],
            axis=1)                                     # (B,L,C)

        emb = self.fc_emb(step_embedding)               # (B,C)
        emb = tf.expand_dims(emb,1)                     # (B,1,C)
        emb = tf.repeat(
            emb,
            repeats=tf.shape(input)[1],
            axis=1)                                     # (B,L,C)
        inp_emb = input + emb                           # (B,L,C)
        conv = self.dil_conv(inp_emb)                   # (B,L,C)
        conv = conv + conditioner
        tanh = tf.keras.activations.tanh(conv)
        sigm = tf.keras.activations.sigmoid(conv)
        o = tanh * sigm
        res_out = self.res_conv(o)
        res_out = input + res_out                             # (B,L,C)
        skip_out = self.skip_conv(o)                          # (B,L,C)
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
        emb = self.dens_emb_1(step_embedding)
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

class ConditionedDiffWaveNet(tf.keras.Model):
    def __init__(self, depth, C, kernel_size):
        super(ConditionedDiffWaveNet,self).__init__()
        self.depth = depth
        self.input_conv = tf.keras.layers.Conv1D(C,1,padding='same',activation='relu')
        self.dens_emb_1 = tf.keras.layers.Dense(512,activation=tf.keras.activations.swish)
        self.dens_emb_2 = tf.keras.layers.Dense(512,activation=tf.keras.activations.swish)
        self.dens_cond_1 = tf.keras.layers.Dense(128,activation=tf.keras.activations.swish)
        self.dens_cond_2 = tf.keras.layers.Dense(128,activation=tf.keras.activations.swish)
        self.res_blocks = []
        for i in range(depth):
            self.res_blocks.append(
                ConditionedResidualLayer(2**i,C,kernel_size)
            )
        self.out_conv1 = tf.keras.layers.Conv1D(C,kernel_size,padding='same',activation='relu')
        self.out_conv2 = tf.keras.layers.Conv1D(1,kernel_size,padding='same')

    def call(self, inputs):
        input = inputs[0]
        step_embedding = inputs[1]
        conditioner = inputs[2]
        input = self.input_conv(input)                             # (B,L,C)
        emb = self.dens_emb_1(step_embedding)
        emb = self.dens_emb_2(emb)
        cond = self.dens_cond_1(conditioner)
        cond = self.dens_cond_2(cond)
        res_input = input
        skip_outputs = tf.TensorArray(
            tf.float32,
            size=self.depth,
            clear_after_read=True)
        for i in range(len(self.res_blocks)):
            res_input,skip_out = self.res_blocks[i]([res_input,emb,cond])
            skip_outputs = skip_outputs.write(i,skip_out)
        out = skip_outputs.stack()                      # (depth,B,L,C)
        out = tf.reduce_sum(out,axis=0)                 # (B,L,C)
        out = self.out_conv1(out)                       # (B,L,C)
        out = self.out_conv2(out)                       # (B,L,1)
        return out

def SimpleResNet(sample_shape, res_block=6):
    input = tf.keras.layers.Input(sample_shape)
    embedding = tf.keras.layers.Input(128)
    emb = tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=1))(embedding)        # (BS,1,128)
    emb = tf.keras.layers.Lambda(lambda x:tf.repeat(x,sample_shape[0],axis=1))(emb)   # (BS,L,128)
    inp = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(input)              # (BS,L,128)
    inp_ = tf.keras.layers.Add()([inp,emb])                                            # (BS,L,128)
    for i in range(res_block):
        o = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(inp_)
        o = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(o)
        o = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(o)
        inp_ = tf.keras.layers.Add()([inp_,o])
    o = tf.keras.layers.Conv1D(1,1,1,'same')(o)
    return tf.keras.Model(inputs=[input,embedding],outputs=o,name="SimpleResNet")

def SimpleRecurrentResNet(sample_shape, res_block=6):
    input = tf.keras.layers.Input(sample_shape)
    embedding = tf.keras.layers.Input(128)
    emb = tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=1))(embedding)        # (BS,1,128)
    emb = tf.keras.layers.Lambda(lambda x:tf.repeat(x,sample_shape[0],axis=1))(emb)   # (BS,L,128)
    inp = tf.keras.layers.LSTM(128,return_sequences=True)(input)                      # (BS,L,128)             
    inp_ = tf.keras.layers.Add()([inp,emb])                                           # (BS,L,128)
    for i in range(res_block):
        o = tf.keras.layers.LSTM(128,return_sequences=True)(inp_)
        o = tf.keras.layers.LSTM(128,return_sequences=True)(o)
        o = tf.keras.layers.LSTM(128,return_sequences=True)(o)
        inp_ = tf.keras.layers.Add()([inp_,o])
    o = tf.keras.layers.LSTM(1,return_sequences=True)(o)
    return tf.keras.Model(inputs=[input,embedding],outputs=o,name="SimpleRecurrentNet")


def DnCNN(sample_shape, nlayers=18):
    input = tf.keras.layers.Input(sample_shape)
    embedding = tf.keras.layers.Input(128)
    emb = tf.keras.layers.Lambda(lambda x:tf.expand_dims(x,axis=1))(embedding)        # (BS,1,128)
    emb = tf.keras.layers.Lambda(lambda x:tf.repeat(x,sample_shape[0],axis=1))(emb)   # (BS,L,128)
    inp = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(input)              # (BS,L,128)
    o= tf.keras.layers.Add()([inp,emb])                                           # (BS,L,128)
    for i in range(nlayers):
        o = tf.keras.layers.Conv1D(128,3,1,'same',activation='relu')(o)
    o = tf.keras.layers.Conv1D(1,1,1,'same')(o)
    return tf.keras.Model(inputs=[input,embedding],outputs=o,name="DnCNN")


