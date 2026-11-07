"""Pure-Python check of the bf16 swap-bound counterexample (float32 math, RNE cast to bf16)."""
import math
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    """Round a float32 value to bfloat16 with round-to-nearest-even (as torch does)."""
    bits = struct.unpack('<I', struct.pack('<f', f32(x)))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


base = [1.0078125, 1.0078125]
source = [1.015625, 1.0078125]
assert all(bf16(v) == v for v in base + source)
w = [f32(1 / math.sqrt(2))] * 2
diff = [f32(s - h) for s, h in zip(source, base)]
coord = f32(f32(w[0] * diff[0]) + f32(w[1] * diff[1]))
edit = [f32(coord * wi) for wi in w]
print('float32 intended edit', edit, 'norm', math.hypot(*edit))
summed32 = [f32(h + e) for h, e in zip(base, edit)]
print('float32 h+edit', summed32, 'exact midpoint?', summed32[0] == 1.01171875)
cast = [bf16(v) for v in summed32]
print('bf16 cast', cast, 'actual edit', [c - h for c, h in zip(cast, base)],
      'norm', math.hypot(*[c - h for c, h in zip(cast, base)]))
# same computation with float64 accumulation, then one rounding to bf16 via float32 nearest
summed64 = [h + e for h, e in zip(base, edit)]
print('float64 h+edit', summed64)
print('float64 -> bf16 directly (round-to-nearest from the exact value):',
      [1.0078125 if v < 1.01171875 else 1.015625 for v in summed64])
print('source-difference norm', math.hypot(*diff))
