## Exact arithmetic validation for "Quantum Observation in Finite Ring Continuum"
## Self-contained; integers only. Reproduces every numbered claim of the paper.
import random
from math import gcd, lcm
random.seed(7)

def order(a, p):
    x, k = 1, 0
    while True:
        x, k = (x * a) % p, k + 1
        if x == 1:
            return k

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

# ---------- Z[i] ledger arithmetic (exact Gaussian integers) ----------
def gm(z, w): return (z[0]*w[0] - z[1]*w[1], z[0]*w[1] + z[1]*w[0])
def ga(z, w): return (z[0]+w[0], z[1]+w[1])
def gc(z):    return (z[0], -z[1])
def gnorm(z): return z[0]*z[0] + z[1]*z[1]
I4 = [(1,0), (0,1), (-1,0), (0,-1)]            # i^l in Z[i]

# ---------- Z[zeta_12] ledger (polys mod x^4 - x^2 + 1) ----------
def zmul(a, b):
    c = [0]*16
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                c[i+j] += ai*bj
    for k in range(15, 3, -1):
        if c[k]:
            c[k-2] += c[k]; c[k-4] -= c[k]; c[k] = 0
    return c[:4]
Z12 = {0: [1,0,0,0]}
for k in range(1, 12):
    Z12[k] = zmul(Z12[k-1], [0,1,0,0])

def shell(p, e, nS, nO):
    N = p - 1
    gS, gO = pow(e, N//nS, p), pow(e, N//nO, p)
    d = gcd(nS, nO)
    HS = {pow(gS, j, p) for j in range(nS)}
    HO = [pow(gO, j, p) for j in range(nO)]
    core = sorted(HS.intersection(HO))
    return N, gS, gO, d, HS, HO, core

# ===================== Carrier F_157, Subject 53, Object 13 =====================
p, e, nS, nO = 157, 5, 52, 12
N, gS, gO, d, HS, HO, core = shell(p, e, nS, nO)
ip = pow(e, N//4, p)                                            # oriented core element
report('157: g=5 primitive', order(e, p) == N)
report('157: core = {1,28,129,156}, |core| = 4 = gcd(52,12)',
       core == [1, 28, 129, 156] and d == 4)
report('157: orientation I = g^{-39} = 28, I^2 = -1',
       pow(e, N - 39, p) == 28 and pow(28, 2, p) == p - 1)
report('157: joint span lcm(52,12) = 156', lcm(nS, nO) == N)

# Fibres: cosets of Q4 in Phi_O; h_alpha = gO^alpha, alpha = 0,1,2
fib = {a: [pow(gO, a + 3*l, p) for l in range(4)] for a in range(3)}
report('157: three fibres partition Phi_O',
       sorted(sum(fib.values(), [])) == sorted(HO))

# Selection rule A(r,s) = 4*[r=s mod 4] over all 52x12 channel pairs
ok = True
for r in range(nS):
    for s in range(nO):
        A = sum(pow(e, ((N//4) * (s - r) * l) % N, p) for l in range(4)) % p
        if A != (4 if (r - s) % 4 == 0 else 0):
            ok = False
report('157: selection rule A(r,s) over all 52x12 channels', ok)

# Proposition (stationary states): psi_s(u) = u^s, drive u -> gO*u
report('157: drive eigenstates U psi_s = gO^s psi_s, all s',
       all(all(pow((gO*u) % p, s, p) == (pow(gO, s, p)*pow(u, s, p)) % p
               for u in HO) for s in range(nO)))

# Measurement vectors v_{r,a} in the Z[i] ledger; random Gaussian-integer state
psi  = {u: (random.randint(-9, 9), random.randint(-9, 9)) for u in HO}
red  = lambda z: (z[0] + ip * z[1]) % p                          # Z[i] -> F_p, i -> 28
psip = {u: red(z) for u, z in psi.items()}

def v(r, a):
    w = {u: (0, 0) for u in HO}
    for l in range(4):
        w[pow(gO, a + 3*l, p)] = I4[(r*l) % 4]
    return w
def pair(x, y):
    s = (0, 0)
    for u in HO:
        s = ga(s, gm(gc(x[u]), y[u]))
    return s
def amp(r, a):
    s = (0, 0)
    for l in range(4):
        s = ga(s, gm(gc(I4[(r*l) % 4]), psi[pow(gO, a + 3*l, p)]))
    return s
def amp_fp(r, a):
    s = 0
    for l in range(4):
        k = pow(ip, l, p)
        s = (s + pow(k, (4 - r) % 4, p) * psip[pow(gO, a + 3*l, p)]) % p
    return s

report('157: basis orthogonality <v,v> = 4*delta*delta',
       all(pair(v(r, a), v(r2, a2)) == ((4, 0) if (r, a) == (r2, a2) else (0, 0))
           for r in range(4) for a in range(3)
           for r2 in range(4) for a2 in range(3)))
lhs = sum(gnorm(amp(r, a)) for r in range(4) for a in range(3))
rhs = 4 * sum(gnorm(z) for z in psi.values())
report('157: Parseval sum |Psi|^2 = 4*||psi||^2  (%d = %d)' % (lhs, rhs), lhs == rhs)
report('157: ledger reduction commutes, Z[i] -> F_157 on all channels',
       all(red(amp(r, a)) == amp_fp(r, a) for r in range(4) for a in range(3)))

def proj(r, a, phi):
    A = (0, 0); vv = v(r, a)
    for l in range(4):
        u = pow(gO, a + 3*l, p)
        A = ga(A, gm(gc(vv[u]), phi[u]))
    return {u: gm(vv[u], A) for u in HO}
phi1 = proj(2, 1, psi); phi2 = proj(2, 1, phi1)
report('157: Lueders idempotence Pi^2 = 4*Pi',
       all(phi2[u] == gm((4, 0), phi1[u]) for u in HO))

# Z[zeta_12] ledger selection rule and its reduction (zeta -> gO = 22)
okL, okR = True, True
for s in range(12):
    for r in range(4):
        for a in range(3):
            S = [0, 0, 0, 0]
            for l in range(4):
                t = zmul(Z12[(s*(a + 3*l)) % 12], Z12[(-3*r*l) % 12])
                S = [S[i] + t[i] for i in range(4)]
            tgt = [4*x for x in Z12[(s*a) % 12]] if (s - r) % 4 == 0 else [0]*4
            if S != tgt:
                okL = False
            Sp = sum(c * pow(gO, k, p) for k, c in enumerate(S)) % p
            Tp = (4 * pow(gO, s*a, p)) % p if (s - r) % 4 == 0 else 0
            if Sp != Tp:
                okR = False
report('157: ledger selection rule Psi = 4*zeta^{s a} iff r = s mod 4', okL)
report('157: ledger reduction matches carrier amplitudes (zeta -> 22)', okR)

# ===================== Carrier F_421: two regimes =====================
p2, e2 = 421, 2
N2 = p2 - 1
report('421: g=2 primitive', order(e2, p2) == N2)
# quantum pair: Subject 61 (C60) reads Object 29 (C28)
_, _, gO2, d2, HS2, HO2, core2 = shell(p2, e2, 60, 28)
report('421: quantum pair core = {1,29,392,420}, gcd = 4, lcm = 420',
       core2 == [1, 29, 392, 420] and d2 == 4 and lcm(60, 28) == 420)
report('421: orientation I = g^{-105} = 2^{-105} = 392, I^2 = -1',
       pow(e2, N2 - 105, p2) == 392 and pow(392, 2, p2) == p2 - 1)
ok = True
for r in range(60):
    for s in range(28):
        A = sum(pow(e2, (105 * (s - r) * l) % N2, p2) for l in range(4)) % p2
        if A != (4 if (r - s) % 4 == 0 else 0):
            ok = False
report('421: selection rule over all 60x28 channels', ok)
report('421: seven outcomes C28/Q4 = C7', 28 // d2 == 7)
# classical pair: Subject 61 (C60) absorbs Object 13 (C12)
_, _, _, d3, HS3, HO3, core3 = shell(p2, e2, 60, 12)
report('421: classical pair gcd(60,12) = 12: total absorption, C1',
       d3 == 12 and set(HO3) <= HS3 and 12 // d3 == 1)

print('all checks passed')
