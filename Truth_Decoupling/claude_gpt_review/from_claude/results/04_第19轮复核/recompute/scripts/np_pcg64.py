"""Pure-Python re-implementation of numpy's default_rng (SeedSequence -> PCG64 XSL-RR) and
Generator.integers(0, n, size) (32-bit Lemire path) plus np.quantile(linear), to re-derive a
percentile bootstrap without numpy.  Self-checks against published numpy outputs."""

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
M128 = (1 << 128) - 1
PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645

INIT_A, MULT_A = 0x43b0d7e5, 0x931e8875
INIT_B, MULT_B = 0x8b51f9dd, 0x58f38ded
MIX_MULT_L, MIX_MULT_R = 0xca01f9dd, 0x4973f715
XSHIFT = 16


def _int_to_u32_words(n):
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n & M32)
        n >>= 32
    return out


class SeedSequence:
    def __init__(self, entropy, pool_size=4):
        self.pool = [0] * pool_size
        ent = _int_to_u32_words(entropy)
        hc = [INIT_A]

        def hashmix(value):
            value = (value ^ hc[0]) & M32
            hc[0] = (hc[0] * MULT_A) & M32
            value = (value * hc[0]) & M32
            value ^= value >> XSHIFT
            return value

        def mix(x, y):
            r = (MIX_MULT_L * x - MIX_MULT_R * y) & M32
            r ^= r >> XSHIFT
            return r

        for i in range(pool_size):
            self.pool[i] = hashmix(ent[i] if i < len(ent) else 0)
        for i_src in range(pool_size):
            for i_dst in range(pool_size):
                if i_src != i_dst:
                    self.pool[i_dst] = mix(self.pool[i_dst], hashmix(self.pool[i_src]))
        for i_src in range(pool_size, len(ent)):
            for i_dst in range(pool_size):
                self.pool[i_dst] = mix(self.pool[i_dst], hashmix(ent[i_src]))

    def generate_state_u64(self, n_words):
        hc = INIT_B
        out32 = []
        for i in range(2 * n_words):
            v = self.pool[i % len(self.pool)]
            v = (v ^ hc) & M32
            hc = (hc * MULT_B) & M32
            v = (v * hc) & M32
            v ^= v >> XSHIFT
            out32.append(v)
        return [out32[2 * i] | (out32[2 * i + 1] << 32) for i in range(n_words)]


class PCG64:
    def __init__(self, seed):
        s = SeedSequence(seed).generate_state_u64(4)
        initstate = (s[0] << 64) | s[1]
        initseq = (s[2] << 64) | s[3]
        self.state = 0
        self.inc = ((initseq << 1) | 1) & M128
        self._step()
        self.state = (self.state + initstate) & M128
        self._step()
        self.has32 = False
        self.u32 = 0

    def _step(self):
        self.state = (self.state * PCG_MULT + self.inc) & M128

    def next64(self):
        self._step()
        st = self.state
        x = ((st >> 64) ^ st) & M64
        rot = st >> 122
        return ((x >> rot) | (x << ((64 - rot) & 63))) & M64

    def next32(self):
        if self.has32:
            self.has32 = False
            return self.u32
        nxt = self.next64()
        self.has32 = True
        self.u32 = nxt >> 32
        return nxt & M32

    def random(self):
        return (self.next64() >> 11) * (1.0 / 9007199254740992.0)

    def integers(self, high, count):
        """Generator.integers(0, high, size=count), int64 path with range < 2**32 (Lemire, unmasked)."""
        rng = high - 1
        assert 0 < rng < M32
        rng_excl = rng + 1
        out = []
        for _ in range(count):
            m = self.next32() * rng_excl
            left = m & M32
            if left < rng_excl:
                thr = (M32 - rng) % rng_excl
                while left < thr:
                    m = self.next32() * rng_excl
                    left = m & M32
            out.append(m >> 32)
        return out


def quantile_linear(values, q):
    a = sorted(values)
    n = len(a)
    pos = q * (n - 1)
    lo = int(pos // 1)
    hi = min(lo + 1, n - 1)
    t = pos - lo
    if t >= 0.5:  # numpy _lerp form
        return a[hi] - (a[hi] - a[lo]) * (1 - t)
    return a[lo] + (a[hi] - a[lo]) * t


def selfcheck():
    r0 = PCG64(0).random()
    r42 = PCG64(42).random()
    i42 = PCG64(42).integers(10, 5)
    i0 = PCG64(0).integers(10, 5)
    print('selfcheck default_rng(0).random() =', repr(r0), 'expected 0.6369616873214543 ->', r0 == 0.6369616873214543)
    print('selfcheck default_rng(42).random() =', repr(r42), 'expected 0.7739560485559633 ->', r42 == 0.7739560485559633)
    print('selfcheck default_rng(42).integers(0,10,5) =', i42, '(remembered numpy output [0, 7, 6, 4, 4])')
    print('selfcheck default_rng(0).integers(0,10,5) =', i0, '(remembered numpy output [8, 6, 5, 2, 3])')


if __name__ == '__main__':
    selfcheck()
