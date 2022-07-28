import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from diffusion.positional_encoder import encode
import numpy as np

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
        

def train(  data:tf.data.Dataset, 
            diffusion_steps:int,
            model:tf.keras.Model,
            opt:tf.keras.optimizers.Optimizer,
            model_name:str,
            step_emb_dim:int=128,
            epochs=100,
            print_every=10):

    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    best_ep_loss = tf.convert_to_tensor(np.inf)
    curr_ep_loss = tf.zeros_like(best_ep_loss)
    
    for ep in range(epochs):
        # sample x_0 from qdata and eps from N_0_1
        for step, x_0 in enumerate(data):
            t = sample_diffusion_step(diffusion_steps)
            eps = sample_gaussian_noise(tf.shape(x_0))
            inp = tf.math.sqrt(alpha_hat[t]) * x_0 + tf.math.sqrt(1. - alpha_hat[t]) * eps
            t_enc = encode(t,step_emb_dim)
            with tf.GradientTape() as tape:
                o = model([inp,t_enc])
                l = tf.norm(eps-o,ord=2)
                l = tf.reduce_mean(l)
            grads = tape.gradient(l, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            curr_ep_loss += l

            # Log every print_every batches.
            if step % print_every == 0:
                tf.print(
                    "Training loss [step: %5d, ep: %5d] = %.4f"
                    % (step,ep, float(l))
                )
        curr_ep_loss = curr_ep_loss / (step + 1)
        tf.print("------------------------------")
        tf.print("EPOCH LOSS: %4f"%(curr_ep_loss))
        if curr_ep_loss < best_ep_loss:
            tf.print("Loss decreased: %4f --> %4f"%(best_ep_loss,curr_ep_loss))
            best_ep_loss = curr_ep_loss
            curr_ep_loss = tf.zeros_like(best_ep_loss)
            model.save(f"ckpt/{model_name}")


def sample(model,shape,diffusion_steps,return_sequence=False,step_emb_dim:int=128,):
    """
    sample from noise, the sample shape should be (bs,seq_len)
    """
    samples = []
    x_t = tf.random.normal(shape)
    beta = variance_schedule(diffusion_steps)
    alpha = get_alpha(beta)
    alpha_hat = get_alpha_hat(alpha)
    beta_hat = get_beta_hat(alpha_hat,beta)
    for i in range(diffusion_steps - 1):
        t = diffusion_steps - 1 - i
        t_enc = encode(t,step_emb_dim)
        eps_theta = model([x_t,t_enc])
        mu = 1./tf.math.sqrt(alpha_hat[t]) * (x_t - (beta[t]/tf.math.sqrt(1.-alpha_hat[t]))*eps_theta)
        sigma = tf.math.sqrt(beta_hat[t])
        dist = MultivariateNormalDiag(loc=mu,scale_identity_multiplier=sigma)
        x_prec = dist.sample()
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
