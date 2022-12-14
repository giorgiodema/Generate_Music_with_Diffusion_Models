import tensorflow as tf
from diffusion.positional_encoder import encode
import numpy as np
from data.dataset import *
from params import params
import subprocess

def variance_schedule(diffusion_steps):
    """
    the betas are set to be linearly increasing constants from 10**-4 to 0.02
    as in https://arxiv.org/abs/2006.11239
    """
    b_0 = 10**-4
    b_T = 0.02
    step_size = (b_T-b_0)/diffusion_steps
    betas = tf.range(start=b_0,limit=b_T,delta=step_size)
    return betas

def sample_diffusion_step(diffusion_steps):
    return tf.random.uniform(
        (1,),
        minval=0,
        maxval=diffusion_steps,
        dtype=tf.dtypes.int32)[0]


def sample_gaussian_noise(shape):
    return tf.random.normal(shape)

def get_alpha(beta):
    return 1. - beta

def get_beta_hat(alpha_hat,beta):
    beta_hat = [beta[0]]
    for i in range(1,alpha_hat.shape[0]):
        bh = beta[i] * (1. - alpha_hat[i-1])/(1. - alpha_hat[i])
        beta_hat.append(bh)
    beta_hat = tf.concat(beta_hat,0)
    return beta_hat

def get_alpha_hat(alpha):
    alpha_hat = []
    t = tf.range(start=0,limit=alpha.shape[0],dtype=tf.float32)
    for i in range(alpha.shape[0]):
        mask = tf.cast(t<=i,tf.float32)
        aux = mask*alpha + (1.-mask)*1.
        alpha_hat.append(tf.reduce_prod(aux))
    alpha_hat = tf.stack(alpha_hat)
    return alpha_hat
        
def forward(x_0,alpha_hat,t):
    eps = sample_gaussian_noise(tf.shape(x_0))
    inp = tf.math.sqrt(alpha_hat[t]) * x_0 + tf.math.sqrt(1. - alpha_hat[t]) * eps
    return inp,eps

@tf.function
def train_step(model,inp,t_enc,eps):
    o = model([inp,t_enc])
    # DEBUG #
    #wav = get_wav(o[0],SR//params["DOWNSAMPLE"])
    #subprocess.run(["ffplay","-"],input=wav.numpy())
    # DEBUG #
    l = tf.norm(eps-o,ord=2)
    l = tf.reduce_mean(l)
    return l

def train(  data:tf.data.Dataset, 
            diffusion_steps:int,
            model:tf.keras.Model,
            opt:tf.keras.optimizers.Optimizer,
            model_name:str,
            step_emb_dim:int=128,
            print_every=10,
            save_every=None,
            ckpt_path="./ckpt",
            resume =False,
            resume_ckpt = None,
            max_steps = 2000000):

    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    best_ep_loss = tf.convert_to_tensor(np.inf)
    curr_ep_loss = tf.zeros_like(best_ep_loss)
    
    step = 0
    it = iter(data)

    if resume:
        model.load_weights(os.path.join(ckpt_path,f"__step_{resume_ckpt}__{model.name}"))
        step = resume_ckpt
    while step<max_steps:
        try:
            # sample x_0 from qdata and eps from N_0_1
            x_0 = next(it)
            t = sample_diffusion_step(diffusion_steps)
            inp,eps = forward(x_0,alpha_hat,t)
            # DEBUG #
            #wav = get_wav(inp[0],SR//params["DOWNSAMPLE"])
            #subprocess.run(["ffplay","-"],input=wav.numpy())
            # DEBUG #
            t_enc = encode(t,step_emb_dim)
            t_enc = tf.expand_dims(t_enc,0)                  
            t_enc = tf.repeat(t_enc,tf.shape(x_0)[0],axis=0) 
            with tf.GradientTape() as tape:
                l = train_step(model,inp,t_enc,eps)
            grads = tape.gradient(l, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            # tf.reduce_sum(tf.concat(list(map(lambda x:tf.abs(tf.reshape(x,-1)),list(filter(lambda x:x!=None,grads)))),axis=0))
            curr_ep_loss += l

            # Log every print_every batches.
            if step % print_every == 0:
                tf.print(
                    "Training loss [step: %10d] = %.4f"
                    % (step, float(l))
                )
                with open(f"log/{model_name}.txt","a") as f:
                    f.write(f"{l},")
            if save_every and step%save_every==0:
                model.save_weights(os.path.join(ckpt_path,f"__step_{step}__{model_name}"))

            step += 1
        except KeyboardInterrupt:
            model.save_weights(os.path.join(ckpt_path,f"__step_{step}__{model_name}"))
        except StopIteration:
            it = iter(data)
            if not save_every:
                model.save_weights(os.path.join(ckpt_path,f"__step_{step}__{model_name}"))

def train_conditioned(  data:tf.data.Dataset,
            diffusion_steps:int,
            model:tf.keras.Model,
            opt:tf.keras.optimizers.Optimizer,
            model_name:str,
            step_emb_dim:int=128,
            class_emb_dim:int=128,
            print_every=10):

    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    best_ep_loss = tf.convert_to_tensor(np.inf)
    curr_ep_loss = tf.zeros_like(best_ep_loss)
    
    step = 0
    it = iter(data)
    while True:
        try:
            # sample x_0 from qdata and eps from N_0_1
            x_0,y = next(it)
            t = sample_diffusion_step(diffusion_steps)
            inp,eps = forward(x_0,alpha_hat,t)
            # DEBUG #
            #wav = get_wav(inp[0],SR//params["DOWNSAMPLE"])
            #subprocess.run(["ffplay","-"],input=wav.numpy())
            # DEBUG #
            t_enc = encode(t,step_emb_dim)
            t_enc = tf.expand_dims(t_enc,0)                  
            t_enc = tf.repeat(t_enc,tf.shape(x_0)[0],axis=0)

            y_enc = tf.map_fn(
                lambda x:encode(x,class_emb_dim),
                tf.cast(y,tf.float32)
            )

            with tf.GradientTape() as tape:
                o = model([inp,t_enc,y_enc])
                # DEBUG #
                #wav = get_wav(o[0],SR//params["DOWNSAMPLE"])
                #subprocess.run(["ffplay","-"],input=wav.numpy())
                # DEBUG #
                l = tf.norm(eps-o,ord=2)
                l = tf.reduce_mean(l)
            grads = tape.gradient(l, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            # tf.reduce_sum(tf.concat(list(map(lambda x:tf.abs(tf.reshape(x,-1)),list(filter(lambda x:x!=None,grads)))),axis=0))
            curr_ep_loss += l

            # Log every print_every batches.
            if step % print_every == 0:
                tf.print(
                    "Training loss [step: %10d] = %.4f"
                    % (step, float(l))
                )
                with open(f"log/{model_name}.txt","a") as f:
                    f.write(f"{l},")

            step += 1
        except KeyboardInterrupt:
            model.save_weights(f"ckpt/__last__{model_name}")
        except StopIteration:
            it = iter(data)
            model.save_weights(f"ckpt/__step_{step}__{model_name}")
    
def train_single_sample(x_0:tf.Tensor, 
            diffusion_steps:int,
            model:tf.keras.Model,
            opt:tf.keras.optimizers.Optimizer,
            model_name:str,
            step_emb_dim:int=128,
            training_steps=10000,
            print_every=10):

    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    best_ep_loss = tf.convert_to_tensor(np.inf)
    curr_ep_loss = tf.zeros_like(best_ep_loss)
    

    for step in range(training_steps):
        t = sample_diffusion_step(diffusion_steps)
        inp,eps = forward(x_0,alpha_hat,t)
        # DEBUG #
        #wav = get_wav(inp[0],SR//params["DOWNSAMPLE"])
        #subprocess.run(["ffplay","-"],input=wav.numpy())
        # DEBUG #
        t_enc = encode(t,step_emb_dim)
        t_enc = tf.expand_dims(t_enc,0)                  
        t_enc = tf.repeat(t_enc,tf.shape(x_0)[0],axis=0) 
        with tf.GradientTape() as tape:
            o = model([inp,t_enc])
            # DEBUG #
            #wav = get_wav(o[0],SR//params["DOWNSAMPLE"])
            #subprocess.run(["ffplay","-"],input=wav.numpy())
            # DEBUG #
            l = tf.norm(eps-o,ord=2)
            l = tf.reduce_mean(l)
        grads = tape.gradient(l, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        # tf.reduce_sum(tf.concat(list(map(lambda x:tf.abs(tf.reshape(x,-1)),list(filter(lambda x:x!=None,grads)))),axis=0))
        curr_ep_loss += l

        # Log every print_every batches.
        if step % print_every == 0:
            tf.print(
                "Training loss [step: %5d/%5d] = %.4f"
                % (step,training_steps, float(l))
            )
            with open(f"test/{model_name}.txt","a") as f:
                f.write(f"{l},")


def backward_process(model,shape,diffusion_steps,return_sequence=False,step_emb_dim:int=128):
    """
    sample from noise, the sample shape should be (bs,seq_len)
    """
    samples = []
    x_t = tf.random.normal(shape)
    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    #beta_hat = get_beta_hat(alpha_hat,beta)
    for i in range(diffusion_steps):
        t = diffusion_steps - 1 - i
        z = tf.random.normal(shape) if t > 0 else \
            tf.zeros(shape)
        t_enc = encode(t,step_emb_dim)
        t_enc = tf.expand_dims(t_enc,0)                  
        t_enc = tf.repeat(t_enc,tf.shape(x_t)[0],axis=0) 
        eps_theta = model([x_t,t_enc])
        #mu = 1./tf.math.sqrt(alpha[t]) * (x_t - (beta[t]/tf.math.sqrt(1.-alpha_hat[t]))*eps_theta)
        #sigma = tf.math.sqrt(beta_hat[t])
        #dist = MultivariateNormalDiag(loc=mu,scale_identity_multiplier=sigma)
        #x_prec = dist.sample()
        x_prec = 1./tf.math.sqrt(alpha[t]) * \
            (x_t - (1. - alpha[t])/(tf.math.sqrt(1.-alpha_hat[t])) * eps_theta ) + \
                 tf.math.sqrt(beta[t])*z
        x_t = x_prec
        if return_sequence:
            samples.append(x_prec)
    if return_sequence:
        return samples
    else:
        return x_prec

def conditioned_backward_process(model,shape,genre,diffusion_steps,return_sequence=False,step_emb_dim:int=128,class_emb_dim:int=128):
    """
    sample from noise, the sample shape should be (bs,seq_len)
    """
    samples = []
    x_t = tf.random.normal(shape)
    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(labels,indices),
        -1
    )
    y = table.lookup(tf.constant(genre))
    if y<0:
        raise ValueError("Invalid genre")
    y_enc = encode(y,class_emb_dim)
    y_enc = tf.expand_dims(y_enc,0)                  
    y_enc = tf.repeat(y_enc,tf.shape(x_t)[0],axis=0) 
    
    for i in range(diffusion_steps):
        t = diffusion_steps - 1 - i
        z = tf.random.normal(shape) if t > 0 else \
            tf.zeros(shape)
        t_enc = encode(t,step_emb_dim)
        t_enc = tf.expand_dims(t_enc,0)                  
        t_enc = tf.repeat(t_enc,tf.shape(x_t)[0],axis=0) 
        eps_theta = model([x_t,t_enc,y_enc])
        #mu = 1./tf.math.sqrt(alpha[t]) * (x_t - (beta[t]/tf.math.sqrt(1.-alpha_hat[t]))*eps_theta)
        #sigma = tf.math.sqrt(beta_hat[t])
        #dist = MultivariateNormalDiag(loc=mu,scale_identity_multiplier=sigma)
        #x_prec = dist.sample()
        x_prec = 1./tf.math.sqrt(alpha[t]) * \
            (x_t - (1. - alpha[t])/(tf.math.sqrt(1.-alpha_hat[t])) * eps_theta ) + \
                 tf.math.sqrt(beta[t])*z
        x_t = x_prec
        if return_sequence:
            samples.append(x_prec)
    if return_sequence:
        return samples
    else:
        return x_prec

def backward_process_from(model,shape,diffusion_steps,x_start,t_start,return_sequence=False,step_emb_dim:int=128):
    """
    sample from noise, the sample shape should be (bs,seq_len)
    """
    samples = []
    x_t = x_start
    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    #beta_hat = get_beta_hat(alpha_hat,beta)
    for i in range(0,t_start+1):
        t = t_start - i
        z = tf.random.normal(shape) if t > 0 else \
            tf.zeros(shape)
        t_enc = encode(t,step_emb_dim)
        t_enc = tf.expand_dims(t_enc,0)                  
        t_enc = tf.repeat(t_enc,tf.shape(x_t)[0],axis=0) 
        eps_theta = model([x_t,t_enc])
        #mu = 1./tf.math.sqrt(alpha[t]) * (x_t - (beta[t]/tf.math.sqrt(1.-alpha_hat[t]))*eps_theta)
        #sigma = tf.math.sqrt(beta_hat[t])
        #dist = MultivariateNormalDiag(loc=mu,scale_identity_multiplier=sigma)
        #x_prec = dist.sample()
        x_prec = 1./tf.math.sqrt(alpha[t]) * \
            (x_t - (1. - alpha[t])/(tf.math.sqrt(1.-alpha_hat[t])) * eps_theta ) + \
                 tf.math.sqrt(beta[t])*z
        x_t = x_prec
        if return_sequence:
            samples.append(x_prec)
    if return_sequence:
        return samples
    else:
        return x_prec


    
if __name__=="__main__":
    beta = variance_schedule(10)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    beta_hat = get_beta_hat(alpha_hat,beta)
    pass
