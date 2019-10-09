#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:55:35 2019

@author: nmei
"""

from glob import glob
import os
import csv
import numpy as np

files = glob('*trials.csv')

for f in files:
    file_name = f.split('.')[0]
    with open(f,'r') as csvin,open('{}.tsv'.format(file_name),'w') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout,delimiter='\t')
        
        for row in csvin:
            tsvout.writerow(row)

files = glob('*trials.tsv')


files = np.sort(files)
files.reshape((-1,9))
