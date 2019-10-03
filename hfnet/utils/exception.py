#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-10-03 00:44
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : exception.py
"""


class EmptyTensorError(Exception):
    pass


class NoGradientError(Exception):
    pass