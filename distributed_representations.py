# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 07:51:54 2020

@author: maste
"""



import numpy as np
from scipy.signal import fftconvolve,deconvolve
import get_wd
from scipy import sparse

import annotated_heat_maps as ahm
import matplotlib.pyplot as plt

def cosine(compare):
    return np.dot(compare,compare.transpose()) / np.outer(np.sqrt(np.sum(compare*compare,1)),np.sqrt(np.sum(compare*compare,1)))


def plot(out,labels,name):
    # plot
    img_size = 15
    fig, ax = plt.subplots()
    im, cbar = ahm.heatmap(out, labels, labels, ax=ax,
                       cmap="gist_heat", cbarlabel='Jaccard Index')
    ahm.annotate_heatmap(im, valfmt="{x:.1f}", fontsize = 10)
    fig.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(img_size, img_size)
    plt.show()
#    fig.savefig('heatmaps/'+name+'.png', dpi=100, transparent = True)

###################
# convolution
    
#in_ = 100
out_ = 1000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],out_))

mem_conv = np.zeros((len(mydict),out_),'complex128')
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(out_,'complex128')
    for i in sparse.find(row)[1]:
        vect *= np.fft.fft(vects[i])
    for i in sparse.find(row)[1]:
        mem_conv[i] += vect

mem_conv = np.real(np.fft.ifft(mem_conv))

first = cosine(mem_conv) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')

###################
# convolution
# localist representation, cast as matrix

#in_ = 500
out_ = 1000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

word_vects = np.diag(np.ones(wd.shape[1]))
#word_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],in_))
rand_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],out_))

mem_conv = np.zeros((len(mydict),out_),'complex128')
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(out_,'complex128')
    for i in sparse.find(row)[1]:
        vect *= np.fft.fft(word_vects[i].dot(rand_vects))
#        row.dot(word_vects).transpose()
    mem_conv += (row.dot(word_vects).transpose()).dot(vect.reshape((1,out_)))
#    for i in sparse.find(row)[1]:
#        mem_conv[i] += vect

mem_conv = np.real(np.fft.ifft(mem_conv))

out = np.zeros((wd.shape[1], out_))
for i in np.arange(word_vects.shape[0]):
    out[i] = word_vects[i].dot(mem_conv)

first = cosine(out) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')


###################
# convolution
# distributed representation, cast as matrix

in_ = 1000
out_ = 10000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

#word_vects = np.diag(np.ones(wd.shape[1]))
word_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],in_))
rand_vects = np.random.normal(0,1/np.sqrt(out_),(in_,out_))

mem_conv = np.zeros((in_,out_),'complex128')
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(out_,'complex128')
    for i in sparse.find(row)[1]:
        vect *= np.fft.fft(word_vects[i].dot(rand_vects))
    
    mem_conv += (row.dot(word_vects).transpose()).dot(vect.reshape((1,out_)))
#    for i in sparse.find(row)[1]:
#        mem_conv[i] += vect

mem_conv = np.real(np.fft.ifft(mem_conv))

out = np.zeros((wd.shape[1], out_))
for i in np.arange(word_vects.shape[0]):
    out[i] = word_vects[i].dot(mem_conv)

first = cosine(out) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')


########################
# multiplication
# localist, matrix

#in_ = 500
out_ = 1000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

word_vects = np.diag(np.ones(wd.shape[1]))
#word_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],in_))
rand_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],out_))

mem_conv = np.zeros((len(mydict),out_))
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(out_)
    for i in sparse.find(row)[1]:
        vect *= word_vects[i].dot(rand_vects)
    mem_conv += (row.dot(word_vects).transpose()).dot(vect.reshape((1,out_)))

out = np.zeros((wd.shape[1], out_))
for i in np.arange(word_vects.shape[0]):
    out[i] = word_vects[i].dot(mem_conv)

first = cosine(out) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')


########################
# multiplication
# localist, matrix
# only mem vector
# doesn't work. I guess there needs to be some static projection matrix

#in_ = 500
out_ = 1000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

word_vects = np.diag(np.ones(wd.shape[1]))
#word_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],in_))
mem = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],out_))

#mem_conv = np.zeros((len(mydict),out_))
for j in np.arange(50):
    
    temp = np.zeros((wd.shape[1],out_))
    for row in wd:
        print(sparse.find(row)[1])
        vect = np.ones(out_)
        for i in sparse.find(row)[1]:
            vect *= word_vects[i].dot(mem)
        temp += (row.dot(word_vects).transpose()).dot(vect.reshape((1,out_)))
    mem += temp

out = np.zeros((wd.shape[1], out_))
for i in np.arange(word_vects.shape[0]):
    out[i] = word_vects[i].dot(mem)

first = cosine(out) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')

########################
# multiplication
# localist, matrix
# with normalization

#in_ = 500
out_ = 1000
wd,mydict = get_wd.loadCorpus('artificialGrammar.txt')

word_vects = np.diag(np.ones(wd.shape[1]))
#word_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],in_))
rand_vects = np.random.normal(0,1/np.sqrt(out_),(wd.shape[1],out_))

mem_conv = np.zeros((len(mydict),out_))+.000000001
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(out_)
    for i in sparse.find(row)[1]:
        vect *= word_vects[i].dot(rand_vects)
    mem_conv += (row.dot(word_vects).transpose()).dot(vect.reshape((1,out_)))
#    mem_conv = mem_conv / np.sqrt((mem_conv*mem_conv).sum(1)).reshape((mem_conv.shape[0],1)) # normalizing with every trial doesn't work
mem_conv = mem_conv / np.sqrt((mem_conv*mem_conv).sum(1)).reshape((mem_conv.shape[0],1)) # batch normalization works

out = np.zeros((wd.shape[1], out_))
for i in np.arange(word_vects.shape[0]):
    out[i] = word_vects[i].dot(mem_conv)

first = cosine(out) 
np.fill_diagonal(first,0)
plot(first,sorted(mydict.keys()),'')

second = cosine(first)
np.fill_diagonal(second,0)
plot(second,sorted(mydict.keys()),'')