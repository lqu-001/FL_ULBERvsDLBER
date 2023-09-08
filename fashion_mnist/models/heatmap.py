#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
def heatmap(R):
    sns.set()
    fig = plt.figure()
    sns_plot = sns.heatmap(R)
    # fig.savefig("heatmap.pdf", bbox_inches='tight') # reduce edge blank
    plt.show()
    
def heatmapro(R):
    sns.set()
    fig = plt.figure()
    sns_plot = sns.heatmap(R,xticklabels=10, yticklabels=10)
    #sns_plot = sns.heatmap(R,cmap='YlGnBu', xticklabels=10, yticklabels=10)
    sns_plot.tick_params(labelsize=5) # heatmap
    # colorbar 
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=15) # colorbar 

    plt.show()
    


