
# coding: utf-8

# In[187]:




# In[10]:

import numpy as np
import theano
import theano.tensor as T
import lasagne
import librosa as lb

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')

import gzip
import pickle
import os
import os.path
import argparse

# Seed for reproduciblity
np.random.seed(42)


# In[188]:

parser = argparse.ArgumentParser(
    description='Music styling',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--subject', type=str, default="02_carnifex_dark_days_myzuka.wav",
                    help='Subject image.')
parser.add_argument('--style', type=str, default="green_day.wav",
                    help='Style image.')
parser.add_argument('--output', type=str, default="result.wav",
                    help='Style image.')
args = parser.parse_args()


# In[84]:

# We need to reshape from a 1D feature vector to a 1 channel 2D image.
# Then we apply 3 convolutional filters with 3x3 kernel size.
l_in = lasagne.layers.InputLayer((None, 257, 173))

l_shape = lasagne.layers.ReshapeLayer(l_in, (-1, 1, 257, 173))

l_conv = lasagne.layers.Conv2DLayer(l_shape, num_filters=1, stride=2, filter_size=3, pad='valid')

l_conv2 = lasagne.layers.Conv2DLayer(l_conv, num_filters=1, stride=2, filter_size=3, pad='valid')

l_conv3 = lasagne.layers.Conv2DLayer(l_conv2, num_filters=1, stride=2, filter_size=3, pad='valid')

l_conv4 = lasagne.layers.Conv2DLayer(l_conv3, num_filters=1, stride=2, filter_size=3, pad='valid')

l_conv5 = lasagne.layers.Conv2DLayer(l_conv4, num_filters=1, stride=2, filter_size=3, pad='valid')

print(lasagne.layers.get_output_shape(l_conv5))
l_out = lasagne.layers.DenseLayer(l_conv5,
                          num_units=2,
                          nonlinearity=lasagne.nonlinearities.softmax)
layers = ['l_conv', 'l_conv2', 'l_conv3', 'l_conv4', 'l_conv5', 'l_out']
net = {}
net['l_conv'] = l_conv2
net['l_conv2'] = l_conv2
net['l_conv3'] = l_conv3
net['l_conv4'] = l_conv4
net['l_conv5'] = l_conv5
net['l_out'] = l_out

layers = {k: net[k] for k in layers}
print(lasagne.layers.get_output_shape(l_conv3))


# In[144]:

import librosa as lb
def f(x):
    return x.real

def g(x):
    return x.imag

f = np.vectorize(f)
g = np.vectorize(g)

def gen_sample(filename, n_batch=10):
    print(filename)
    src, sr = lb.load (filename, sr=11025)
    frame_size = sr*4
    res = []
    idx = np.random.choice(len(src) - frame_size, n_batch)
    for pos in idx:
        frame = src[pos: pos + frame_size]
        SRC = lb.stft (frame, n_fft=512, hop_length=256)
        res.append(f(SRC))
    return res

def get_samples(filename, offset=0):
    src, sr = lb.load (filename, sr=11025)
    frame_size = sr*4
    res = []
    for pos in range(0, len(src), frame_size):
        frame = src[pos: pos + frame_size]
        SRC = lb.stft (frame, n_fft=512, hop_length=256)
        res.append((f(SRC), g(SRC)))
    return res

def restore_sample(aSRC, pSRC, filename='test.wav'):
    signal = lb.istft(aSRC + pSRC, hop_length=256)
    
    print(len(signal))
    return signal
    
def save_sample(filename, y):
    print(filename)
    lb.output.write_wav(filename, y, 11025)


# In[180]:

values = pickle.load(open('weights0.500.pkl', 'rb'))
lasagne.layers.set_all_param_values(layers.values(), values)


# In[116]:

def gram_matrix(x):
    x = x.flatten(ndim=3)
    g = T.tensordot(x, x, axes=([2], [2]))
    return g


def content_loss(P, X, layer):
    p = P[layer]
    x = X[layer]
    
    loss = 1./2 * ((x - p)**2).sum()
    return loss


def style_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    A = gram_matrix(a)
    G = gram_matrix(x)
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    loss = 1./(4 * N**2 * M**2) * ((G - A)**2).sum()
    return loss

def total_variation_loss(x):
    return (((x[:,:-1,:-1] - x[:,1:,:-1])**2 + (x[:,:-1,:-1] - x[:,:-1,1:])**2)**1.25).sum()


# In[ ]:

target_samples = get_samples(args.subject)
style_samples = get_samples(args.style)
style_music_array = [x[0] for x in style_samples]
target_music_array = [x[0] for x in target_samples]
samples_cnt = min(len(style_music_array), len(target_music_array))


# In[16]:




# In[127]:

input_im_theano = T.tensor3()
outputs = lasagne.layers.get_output(layers.values(), input_im_theano)
output_genre = lasagne.layers.get_output(l_out, input_im_theano)
target_features, style_features = None, None

def set_features(target_music, style_music):
    global target_features
    global style_features
    target_features = {k: theano.shared(output.eval({input_im_theano: np.array([target_music])}))
                      for k, output in zip(layers.keys(), outputs)}
    style_features = {k: theano.shared(output.eval({input_im_theano: np.array([style_music])}))
                    for k, output in zip(layers.keys(), outputs)}


# In[128]:

from lasagne.utils import floatX
generated_music = theano.shared(floatX(np.random.uniform(-128, 128, (1, 257, 173))))

gen_features = lasagne.layers.get_output(layers.values(), generated_music)
gen_features = {k: v for k, v in zip(layers.keys(), gen_features)}


# In[185]:

eval_loss, eval_grad = None, None 
def set_loss():
    # Define loss function
    losses = []

    # content loss
    losses.append(0.2e-3 * content_loss(target_features, gen_features, 'l_conv5'))

    # style loss
#     losses.append(0.2e5 * style_loss(style_features, gen_features, 'l_conv'))
    losses.append(0.2e6 * style_loss(style_features, gen_features, 'l_conv2'))
    losses.append(0.2e7 * style_loss(style_features, gen_features, 'l_conv3'))
    losses.append(0.2e7 * style_loss(style_features, gen_features, 'l_conv4'))
    
    # total variation penalty
    losses.append(0.1e-7 * total_variation_loss(generated_music))


#     print(type(output_genre))
#     loss = T.mean(lasagne.objectives.categorical_crossentropy(output_genre, lasagne.layers.get_all_params(l_out)))
#     losses.append(loss)
    total_loss = sum(losses)
    grad = T.grad(total_loss, generated_music)
    global f_loss
    global f_grad
    f_loss = theano.function([], total_loss)
    f_grad = theano.function([], grad)
    
    global eval_loss
    global eval_grad

    # Helper functions to interface with scipy.optimize
    def eval_loss(x0):
        global f_loss
        x0 = floatX(x0.reshape((1, 257, 173)))
        generated_music.set_value(x0)
        return f_loss().astype('float64')

    def eval_grad(x0):
        x0 = floatX(x0.reshape((1, 257, 173)))
        generated_music.set_value(x0)
        global f_grad
        return np.array(f_grad()).flatten().astype('float64')


# In[79]:




# In[80]:

# print(xs[-1].shape)


# In[186]:

import scipy.optimize
xss = []
for j in range(min(15, samples_cnt)):
    set_features(target_music_array[j], style_music_array[j])
    set_loss()
    # Initialize with a noise image
    generated_music.set_value(floatX(np.random.uniform(-128, 128, (1, 257, 173))))

    x0 = generated_music.get_value().astype('float64')
    xs = []
    xs.append(x0)

    # Optimize, saving the result periodically
    for i in range(8):
        print(i)
        scipy.optimize.fmin_l_bfgs_b(eval_loss, x0.flatten(), fprime=eval_grad, maxfun=100)
        x0 = generated_music.get_value().astype('float64')
        xs.append(x0)
    xss.append(restore_sample(xs[-1][0], target_samples[j][1]))
    print(len(xss))


# In[92]:

xss


# In[174]:

save_sample(args.output, np.concatenate(xss))


# In[ ]:




# In[ ]:



