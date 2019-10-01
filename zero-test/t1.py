#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-01 15:37
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : t1.py
"""
a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}


import pickle

# pickle a variable to a file
file = open('pdict.pkl', 'wb')
pickle.dump(a_dict, file)
file.close()

# reload a file to a variable
with open('pdict.pkl', 'rb') as file:
    p_dict = pickle.load(file)

print(p_dict)


import joblib

with open('pdict.pkl', 'rb') as file:
    j_dict = joblib.load(file)

print(j_dict)


print(j_dict == p_dict)


