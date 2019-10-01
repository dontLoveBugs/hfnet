#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-01 15:41
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : t2.py
"""

a_dict = {'da': 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}

import pickle
import joblib

# pickle a variable to a file
file = open('jdict.pkl', 'wb')
joblib.dump(a_dict, file)
file.close()

# reload a file to a variable
with open('jdict.pkl', 'rb') as file:
    j_dict = joblib.load(file)

print(j_dict)

with open('jdict.pkl', 'rb') as file:
    p_dict = pickle.load(file)

print(p_dict)

print(j_dict == p_dict)
