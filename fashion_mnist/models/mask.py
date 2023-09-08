#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import scipy.sparse as spy
import numpy as np

def mask(row,col,density,seedval):
    matrixformat='coo'
    D=spy.rand(row,col,density=density,format=matrixformat,dtype=np.dtype('float32'),random_state=np.random.seed(seedval))
    D=D.todense()
    E=np.where(D!=0, 1, D) 
    return E
    
def topk(M,sparsity):
    import heapq
    import numpy as np
    A=np.abs(M)
    B=A.flatten()
    num=int((np.size(B))*sparsity)
    maxseq=heapq.nlargest(num,B)
    absthre=min(maxseq)
    C=((M<absthre)&(M>-(absthre)))
    M[C]=0
    return M
    

def rank(a):
    idx = list(range(len(a)))
    idx.sort(key=lambda x: a[x])  # rank a, return index

    cores_rank = [0] * len(idx)      # Creates empty list of indices
    for i, value in enumerate(idx):
        cores_rank[value] = i
    return cores_rank
    
def entropy(x):
    x=x.flatten()
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent

def gentropy(inx):
    import torch
    inx = abs(inx)
    entro = torch.sum(inx*torch.log2(inx))
    return abs(entro)
    
'''def quandequan(indata, bitlength):
    import numpy as np
    import torch
    import torch.nn.functional as F
    indata=torch.tensor(indata)
    factor=(indata.max()-indata.min())/(2**bitlength-1)
    quan=F.hardtanh(torch.round(indata/factor), min_val=-2**(bitlength-1), max_val=2**(bitlength-1)-1)
    dequan=quan*factor 
    return dequan
    
def quandequan_positive(indata, bitlength):
    import numpy as np
    import torch
    import torch.nn.functional as F
    indata=torch.tensor(indata)
    factor=(indata.max()-indata.min())/(2**bitlength-1)
    quan=F.hardtanh(torch.round(indata/factor), min_val=0, max_val=2**bitlength -1)
    dequan=quan*factor 
    return dequan'''
    
def quandequan_range(indata, bitlength): #my target traditional quantizer
    import torch
    factor=(indata.max()-indata.min())/(2**bitlength-1)
    quan=torch.round((indata-indata.min())/factor)
    dequan=quan*factor + indata.min()
    return dequan

  
def quandequan_norm(indata, bit): #stochastic uniform quantization
    import torch
    bins=torch.round((2**bit -1)*torch.abs(indata)/torch.norm(indata))
    dequan=torch.div(bins,2**bit -1)*torch.sign(indata)*torch.norm(indata) 
    return dequan
    
def rang(x):
    dis=np.max(x)-np.min(x)
    return dis

#--------------comm and comp-----------------------------

import struct
def binary(num):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

import numpy as np
def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise    

#--------encoder and decoder, good--------------    
import struct
def fp32tobin(num):
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


def bintofp32(symbol):
    S=symbol[0]; E=symbol[1:9]; M=symbol[9:32]
    oneplusM='1'+M
    E_d=0
    for i in range (8):
        E_d+=int(E[i])*2**(7-i)
    shift=E_d-127
    Z=0;
    for i in range (24):
        Z+=int(oneplusM[i])*2**(shift-i)
    target=(Z)*(-1)**(int(S))
    return target 


'''#------complete by def Nan, inf    
def bintofp32(symbol):
    S=symbol[0]; E=symbol[1:9]; M=symbol[9:32]
    if (E=='11111111'):
        if (M!='00000000000000000000000'):
            target=float('nan')  #NaN
        else:
            target=(-1)**(int(S))*float('inf') #+_inf
    elif (E=='00000000'):             #not standard code   
        oneplusM='0'+M
        Z=0;
        for i in range (24):
            Z+=int(oneplusM[i])*2**(-126-i)
        target=(Z)*(-1)**(int(S))
    else:        
        oneplusM='1'+M
        E_d=0
        for i in range (8):
            E_d+=int(E[i])*2**(7-i)
        shift=E_d-127
        Z=0;
        for i in range (24):
            Z+=int(oneplusM[i])*2**(shift-i)
        target=(Z)*(-1)**(int(S))   
    return target '''
    
#-------------given bitlength, convert fp32 into binary sequence
import torch
def quan_bit(intensor, bitlength):
    outseq = []
    factor = (intensor.max()-intensor.min())/(2**bitlength-1)
    for item in intensor:
        number = torch.round((item-intensor.min())/factor)
        bi = bin(int(number))[2:].rjust(bitlength,'0')
        outseq.append(bi)
    return outseq, factor, intensor.min()

def dequan_bit(inseq, factor, mini, bitlength):
    outfp32 = []
    for item in inseq:
        number = 0
        for i in range(bitlength):
            number += int(item[i])*2**(bitlength-1-i)
        dequan = number*factor + mini
        outfp32.append(dequan)
    return outfp32

#fp32quant2number, number2binary; binary2number, number2fp32
def int2bin(x, bitlength):
    binary_str = bin(x)[2:]
    padded_str = binary_str.zfill(bitlength)
    binary_tensor = torch.tensor([int(bit) for bit in padded_str])
    return binary_tensor
def quan_bit1(intensor, bitlength):
    factor = (intensor.max()-intensor.min())/(2**bitlength-1)   
    a1d=intensor.view(-1)
    a_binary = torch.empty((1, intensor.numel(), bitlength), dtype=torch.int)
    for i in range(intensor.numel()):
        number = int(torch.round((a1d[i]-intensor.min())/factor))
        binary_x = int2bin(number, bitlength)
        a_binary[0,i] = binary_x
    #a_binary = a_binary.view(intensor.shape + (bitlength,))
    return a_binary,factor,intensor.min() #output(1,x,bitlength)
    
def bin2int(binary_tensor):
    binary_str = ''.join([str(bit.item()) for bit in binary_tensor])
    int_val = int(binary_str, 2)
    return int_val
def dequan_bit1(intensor, a_binary, factor, mini):
    noisy_a=torch.empty((1,intensor.numel()),dtype=torch.float)
    for i in range(intensor.numel()):
        noisy_a[0,i]=bin2int(a_binary[0,i,:])*factor+mini
    noisy_a=noisy_a.view(intensor.shape)
    return noisy_a  
#-------------histogram--------------------



