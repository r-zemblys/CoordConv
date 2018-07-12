#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:43:17 2018

@author: raimondas
"""


import torch
from CoordConv import CoordConv


##square input 
batch, channels, xdim, ydim = 1, 1, 5, 5
data = torch.randn(batch, channels, xdim, ydim)
model = CoordConv(xdim, ydim, False, channels+2, 2, 3)
output_sq = model(data)

##rectangle input
batch, channels, xdim, ydim = 1, 1, 3, 5
data = torch.randn(batch, channels, xdim, ydim)
model = CoordConv(xdim, ydim, False, channels+2, 2, 3)
output_rec = model(data)

