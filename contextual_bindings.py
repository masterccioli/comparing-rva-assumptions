# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:00:45 2020

@author: maste
"""

import numpy as np
from scipy.signal import fftconvolve,deconvolve
import get_wd
from scipy import sparse

# generate random vectors for each word
dims = 10000
wd,mydict = get_wd.loadCorpus('../../first_second_order/corpus/artificialGrammar.txt')
vects = np.random.normal(0,1/np.sqrt(dims),(len(mydict),dims))

def cosine_table_self(vects): # get cosine table, input one matrix
    return vects.dot(vects.transpose()) / \
            np.outer(np.sqrt(np.power(vects, 2).sum(1)),
                     np.sqrt(np.power(vects,2).sum(1)))
            
        
def cosine_table(vects_a,vects_b): # get cosine sims between two matrices
    return vects_a.dot(vects_b.transpose()) / \
            np.outer(np.sqrt(np.power(vects_a,2).sum(1)),
                     np.sqrt(np.power(vects_b,2).sum(1)))

cosine_table_self(vects)

# input layer is a localist word activation
# output layer is a distributed activation
# where the distribution is the sum of convolved vectors where a word occurs
mem_conv = np.zeros((len(mydict),dims),'complex128')
for row in wd:
    print(sparse.find(row)[1])
    vect = np.ones(dims,'complex128')
    for i in sparse.find(row)[1]:
        vect *= np.fft.fft(vects[i])
    for i in sparse.find(row)[1]:
        mem_conv[i] += vect

mem_conv = np.fft.irfft(mem_conv)

first_conv = cosine_table_self(mem_conv)
np.fill_diagonal(first_conv,0)

second_conv = cosine_table_self(first_conv)

mem_sum = np.zeros((len(mydict),dims))
for row in wd:
    print(sparse.find(row)[1])
    for i in sparse.find(row)[1]:
        mem_sum[i] += vects[i]

first_sum = cosine_table_self(mem_sum)
np.fill_diagonal(first_sum,0)

second_sum = cosine_table_self(first_sum)
