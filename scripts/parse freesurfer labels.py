#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 12:52:37 2019

@author: nmei
"""

url = 'https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT'

from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


response = requests.get(url)
soup = BeautifulSoup(response.text, 'lxml')
text = soup.find('pre').find(text=True)
column_name = ['#No.','Label Name','R','G','B','A']

df = {name:[] for name in column_name}

for chunk in text.split('\n\n')[2:]:
    for row in chunk.split('\n'):
        if len(row) > 0:
            needed = re.findall('\S+',row)
            if "#" in row:
                print(needed)
            else:
                for name,v in zip(column_name,needed):
                    df[name].append(v)

df = pd.DataFrame(df)
df = df[column_name]

df.to_csv('FreesurferLTU.csv',index=False)









