"""Read float32 tensors from a torch.save zip file WITHOUT torch and without executing arbitrary pickle code.

The unpickler whitelists exactly three globals and maps them to inert local stand-ins; any other global
raises.  Storage bytes are read directly from the zip member and decoded with struct (little-endian float32)."""
import collections
import io
import math
import pickle
import struct
import zipfile


class _Tensor:
    __slots__ = ('storage', 'offset', 'size', 'stride')

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
    return _Tensor(storage, storage_offset, size, stride)


class _FloatStorageTag:
    dtype = 'float32'


class _Restricted(pickle.Unpickler):
    ALLOWED = {('torch._utils', '_rebuild_tensor_v2'): _rebuild_tensor_v2,
               ('torch', 'FloatStorage'): _FloatStorageTag,
               ('collections', 'OrderedDict'): collections.OrderedDict}

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f'blocked global {module}.{name}')

    def persistent_load(self, pid):
        assert isinstance(pid, tuple) and pid[0] == 'storage' and pid[1] is _FloatStorageTag, pid
        return {'key': str(pid[2]), 'numel': int(pid[4])}


def load_pt(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    root = names[0].split('/')[0]
    assert z.read(f'{root}/byteorder') == b'little'
    obj = _Restricted(io.BytesIO(z.read(f'{root}/data.pkl'))).load()

    def materialise(o):
        if isinstance(o, _Tensor):
            raw = z.read(f'{root}/data/{o.storage["key"]}')
            vals = struct.unpack(f'<{len(raw) // 4}f', raw)
            assert len(o.size) == 1 and o.stride == (1,), (o.size, o.stride)
            return {'values': list(vals[o.offset:o.offset + o.size[0]]), 'raw': raw[4 * o.offset:4 * (o.offset + o.size[0])]}
        if isinstance(o, dict):
            return {k: materialise(v) for k, v in o.items()}
        if isinstance(o, list):
            return [materialise(v) for v in o]
        return o

    return materialise(obj)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cos(a, b):
    return dot(a, b) / (norm(a) * norm(b))
