"""Small read-only validation helpers for future R19 reproduction checks."""
import math
from pathlib import Path
import torch


def checkpoint(path):
    path=Path(path)
    try:
        value=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise ValueError('restricted checkpoint loading failed: '+str(path)) from exc
    if not isinstance(value,dict):raise ValueError('checkpoint must be a mapping: '+str(path))
    return value


def vector(value,name,hidden=None):
    if not isinstance(value,torch.Tensor) or value.ndim!=1 or not value.is_floating_point() or value.numel()==0:
        raise ValueError(name+': expected a nonempty one-dimensional real floating tensor')
    if hidden is not None and value.numel()!=hidden:raise ValueError(name+': hidden size differs')
    if not bool(torch.isfinite(value).all()) or float(value.abs().max())==0:
        raise ValueError(name+': nonfinite or zero direction')
    return value


def layer_vectors(adapter,field,path):
    values=adapter.get(field)
    if not isinstance(values,dict) or not values:raise ValueError(str(path)+': missing/empty '+field)
    result={};hidden=adapter.get('hidden_size')
    if hidden is not None and (type(hidden) is not int or hidden<1):raise ValueError('invalid hidden size')
    for key,value in values.items():
        if isinstance(key,bool) or not isinstance(key,(int,str)):
            raise ValueError('invalid layer key')
        try:layer=int(key)
        except (ValueError,TypeError):raise ValueError('invalid layer key') from None
        if layer<0 or (isinstance(key,str) and str(layer)!=key) or layer in result:
            raise ValueError('duplicate/noncanonical layer key')
        result[layer]=vector(value,str(path)+'/'+field+'/'+str(key),hidden)
    return result


def cosine(a,b):
    a=vector(a,'first vector');b=vector(b,'second vector')
    if a.shape!=b.shape:raise ValueError('cosine vectors differ in shape')
    # Scale before norm so very large/small finite directions do not overflow.
    x=a.double();y=b.double();x=x/x.abs().max();y=y/y.abs().max()
    x=x/torch.sqrt((x*x).sum());y=y/torch.sqrt((y*y).sum())
    value=float((x*y).sum())
    if not math.isfinite(value) or abs(value)>1+1e-10:raise ValueError('invalid cosine')
    return max(-1.,min(1.,value))
