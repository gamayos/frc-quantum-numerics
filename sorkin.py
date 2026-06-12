## T1: Sorkin hierarchy in the FRC ledger.
## Claims verified by exact integer arithmetic (no floats):
##   (S1) second-order interference is generically nonzero (the theory interferes);
##   (S2) third- and fourth-order interference vanish identically on ledger states,
##        for every channel, outcome, and random state -- the Sorkin nullity;
##   (S3) sub-horizon (d*||psi||^2 < p) the carrier shadow determines the exact
##        counts, so the observed Sorkin combination is exactly zero;
##   (S4) super-window, the observed (lifted-shadow) Sorkin combination is always
##        an integer multiple of p: wrap deviations are quantised at the horizon,
##        and small nonzero kappa is forbidden at every scale.
import random
from math import gcd
random.seed(11)

def gm(z, w): return (z[0]*w[0] - z[1]*w[1], z[0]*w[1] + z[1]*w[0])
def ga(z, w): return (z[0]+w[0], z[1]+w[1])
def gc(z):    return (z[0], -z[1])
def gnorm(z): return z[0]*z[0] + z[1]*z[1]
I4 = [(1,0), (0,1), (-1,0), (0,-1)]

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

def subsets(slits):
    out = []
    for mask in range(1, 1 << len(slits)):
        out.append(tuple(i for i in range(len(slits)) if mask >> i & 1))
    return out

def sorkin_run(p, e, nO, d, nslits, coeff):
    """Random nslits Z[i] components on the object cycle; full Sorkin hierarchy.
    coeff: callable returning one random Gaussian-integer coefficient."""
    N = p - 1
    gO = pow(e, N//nO, p)
    HO = [pow(gO, j, p) for j in range(nO)]
    q  = nO // d                                    # number of outcomes
    coreg = nO // d                                 # fibres indexed mod nO/d
    # pointer characters on the core ~ Z_d via i-powers only valid for d=4:
    assert d == 4
    slits = []
    for _ in range(nslits):
        slits.append({u: coeff() for u in HO})
    def amp(S, r, a):
        s = (0, 0)
        for l in range(4):
            u = pow(gO, a + coreg*l, p)
            comp = (0, 0)
            for i in S:
                comp = ga(comp, slits[i][u])
            s = ga(s, gm(gc(I4[(r*l) % 4]), comp))
        return s
    W = {}
    for S in subsets(range(nslits)):
        for r in range(4):
            for a in range(q):
                W[(S, r, a)] = gnorm(amp(S, r, a))
    return W, q

def hier(W, q, k, nslits):
    """k-th order Sorkin combination for the first k slits, all channels/outcomes."""
    from itertools import combinations
    vals = []
    base = tuple(range(k))
    for r in range(4):
        for a in range(q):
            tot = 0
            for m in range(1, k+1):
                sgn = (-1)**(k-m)
                for c in combinations(base, m):
                    tot += sgn * W[(tuple(sorted(c)), r, a)]
            vals.append(tot)
    return vals

wide   = lambda: (random.randint(-9, 9), random.randint(-9, 9))
binary = lambda: (random.randint(0, 1), 0)
deep   = lambda: (7*random.randint(-9, 9), 7*random.randint(-9, 9))

for (p, e, nO, tag) in [(157, 5, 12, '157/53/13'), (421, 2, 28, '421/61/29')]:
    W, q = sorkin_run(p, e, nO, 4, 4, coeff=wide)
    I2 = hier(W, q, 2, 4); I3 = hier(W, q, 3, 4); I4v = hier(W, q, 4, 4)
    report('%s: I2 generically nonzero (max |I2| = %d)' % (tag, max(map(abs, I2))),
           any(v != 0 for v in I2))
    report('%s: I3 = 0 exactly, all channels and outcomes' % tag,
           all(v == 0 for v in I3))
    report('%s: I4 = 0 exactly, all channels and outcomes' % tag,
           all(v == 0 for v in I4v))

# (S3)/(S4): shadow readout. Lift L(w) = (w mod p); observed Sorkin combination
# from lifted shadows. Sub-horizon: zero exactly. Super-window: multiple of p.
p, e, nO = 157, 5, 12
W, q = sorkin_run(p, e, nO, 4, 3, coeff=binary)    # 0/1 coefficients: w <= 144 < p
sub = all(w < p for w in W.values())
I3hat = []
from itertools import combinations
for r in range(4):
    for a in range(q):
        tot = 0
        for m in range(1, 4):
            for c in combinations((0, 1, 2), m):
                tot += (-1)**(3-m) * (W[(tuple(sorted(c)), r, a)] % p)
        I3hat.append(tot)
report('157: sub-horizon tallies (all w < p): observed I3-hat = 0 exactly',
       sub and all(v == 0 for v in I3hat))

W, q = sorkin_run(p, e, nO, 4, 3, coeff=deep)       # forced super-window tallies
big = any(w >= p for w in W.values())
I3hat, nz = [], 0
for r in range(4):
    for a in range(q):
        tot = 0
        for m in range(1, 4):
            for c in combinations((0, 1, 2), m):
                tot += (-1)**(3-m) * (W[(tuple(sorted(c)), r, a)] % p)
        I3hat.append(tot)
        if tot != 0: nz += 1
report('157: super-window tallies present and wrap quantisation I3-hat in pZ '
       '(nonzero cases: %d, all multiples of %d)' % (nz, p),
       big and all(v % p == 0 for v in I3hat) and nz > 0)

print('sorkin: all checks passed')
