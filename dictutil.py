# Copyright 2013 Philip N. Klein
def dict2list(dct, keylist): return [dct[a] for a in keylist]

def list2dict(L, keylist): return {a:b for (a,b) in zip(keylist, L)}

def listrange2dict(L): return list2dict(L, list(range(len(L))))