#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Usuário'
import sys
import os


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_files_dir(_dir, ext='.base', pre='u'):
    fd = dict()
    for f in os.listdir(_dir):
        if f.endswith(ext):
            # fl.append(_dir+'/'+f)
            tmp = f.rstrip(ext).lstrip(pre)
            if is_number(tmp):
                fd[tmp] = _dir+'/'+f
    return fd


def folder2pairs(_dir, extl='.base', extt='.test', pre='u'):
    fdl = get_files_dir(_dir, ext=extl, pre=pre)
    fdt = get_files_dir(_dir, ext=extt, pre=pre)
    setl = set(fdl.keys())
    sett = set(fdt.keys())

    for f in setl.symmetric_difference(sett):
        if f in fdl.keys():
            print "1.No match found for: {}".format(fdl[f])
        else:
            print "2.No match found for: {}".format(fdt[f])

    pairs = []
    for f in sorted(setl.intersection(sett)):
        pairs.append((fdl[f], fdt[f], f))

    return pairs