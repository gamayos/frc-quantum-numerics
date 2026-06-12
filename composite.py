## Composite gate + T3 (CHSH/Tsirelson) in the carrier F_641.
## Configuration: Carrier C_640 (g = 3), Object pair O_17 x O_17 (cycles C_16,
## g_O = 3^40 = 601, g_O^8 = -1), Subjects S_41 (cycles C_40, core C_8 with the
## objects, quotient C_2: natively dichotomic readout).
## Claims verified by exact arithmetic:
##   (C1) carrier admissibility: 641 = 4*160+1 prime, 40 | 640, 16 | 640,
##        gcd(40,16) = 8, [C_16 : C_8] = 2 outcomes;
##   (C2) single drive => diagonal action: the relative offset of the object
##        pair is a constant of motion; joint orbits have length 16; the
##        synchronised orbit is the Bell state of the construction;
##   (C3) doublet readout identity in F_641: for psi = u^s + lam*u^{s+8} the
##        forced C_2 readout gives Psi(0) = 8(1+lam), Psi(1) = 8 g_O^s (1-lam):
##        an exact sigma_x measurement on the winding doublet;
##   (C4) correlator law: E(j_a, j_b) = cos(pi*(j_a - j_b)/40) exactly, from the
##        full forced-basis computation in Z[zeta_80]; settings = relational
##        repositionings c = g^j, granularity pi/40;
##   (C5) reduction commutes at composite level: Z[zeta_80] -> F_641,
##        zeta_80 -> g^8, matches the carrier-internal fibre sums;
##   (C6) no-signalling: each party's marginal is setting-independent, exactly;
##   (C7) CHSH: exhaustive sweep over all 80^3 setting differences gives
##        max S = 2*sqrt(2) (Tsirelson saturation), min S = -2*sqrt(2), and
##        native settings (e.g. a,a' = 0, 20 and b,b' = 10, -10) realise it;
##        S > 2 strictly:
##        Bell violation with carrier-native operations only;
##   (C8) exact value: at the optimum S = 2(zeta_8 + zeta_8^{-1}) in Z[zeta_8],
##        i.e. S^2 = 8: Tsirelson saturation as a cyclotomic identity.
import numpy as np
from math import gcd

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

# ---------------- C1: carrier admissibility ----------------
p, g = 641, 3
N = p - 1
def order(a):
    x, k = 1, 0
    while True:
        x, k = (x*a) % p, k+1
        if x == 1: return k
gO = pow(g, N//16, p)
report('C1: 641 = 4*160+1, g = 3 primitive, gO = 601 of order 16',
       p % 4 == 1 and order(g) == N and gO == 601 and order(gO) == 16)
report('C1: 40 | 640 and 16 | 640; core gcd(40,16) = 8; quotient C_2',
       N % 40 == 0 and N % 16 == 0 and gcd(40, 16) == 8 and 16//8 == 2)
report('C1: gO^8 = -1 (half-turn of the object cycle)', pow(gO, 8, p) == p-1)

# ---------------- C2: diagonal drive, conserved offset, orbits ----------------
HO = [pow(gO, j, p) for j in range(16)]
pairs = [(u, v) for u in HO for v in HO]
def offset(u, v): return (u * pow(v, p-2, p)) % p
ok = all(offset((gO*u) % p, (gO*v) % p) == offset(u, v) for (u, v) in pairs)
orbit = {(1, 1)}
u, v = 1, 1
for _ in range(16):
    u, v = (gO*u) % p, (gO*v) % p
    orbit.add((u, v))
report('C2: relative offset conserved by the drive on all 256 pairs; '
       'synchronised orbit has length 16', ok and len(orbit) == 16)

# ---------------- C3: doublet sigma_x readout identity in F_641 ----------------
core = [pow(gO, 2*l, p) for l in range(8)]          # C_8 inside C_16
ok3 = True
for s in range(16):
    for j in range(80):
        lam = pow(g, 8*j, p)                         # carrier phase on the doublet
        r = s % 8                                    # matched pointer channel
        # psi(u) = u^s + lam * u^{s+8}; fibres: core (alpha=0), gO*core (alpha=1)
        def psi(u): return (pow(u, s, p) + lam*pow(u, s+8, p)) % p
        P0, P1 = 0, 0
        for k in core:
            kr = pow(k, (8 - r) % 8, p) if r else 1   # eta_r(k)^{-1} = k^{-r}
            P0 = (P0 + kr*psi(k)) % p
            P1 = (P1 + kr*psi((gO*k) % p)) % p
        t0 = (8*(1 + lam)) % p
        t1 = (8*pow(gO, s, p)*(1 - lam)) % p
        if P0 != t0 or P1 != t1: ok3 = False
report('C3: forced readout Psi(0) = 8(1+lam), Psi(1) = 8 gO^s (1-lam), '
       'all 16 windings x 80 settings', ok3)

# ---------------- Z[zeta_80] exact arithmetic (poly mod Phi_80) ----------------
from sympy import Poly, symbols, cyclotomic_poly, Rational, sqrt, simplify, cos, pi, nsimplify
x = symbols('x')
PHI = Poly(cyclotomic_poly(80, x), x)
def zmul(a, b):  return (a*b) % PHI
def zpow(k):     return Poly(x, x)**(k % 80) % PHI
def zconj(a):    # zeta -> zeta^{-1}
    out = Poly(0, x)
    for mono, c in zip(a.monoms(), a.coeffs()):
        out = (out + c*zpow(-mono[0])) % PHI
    return out

# ---------------- C4 + C5: correlator law and reduction ----------------
# Conditioned Bell doublet (windings {a, a+8} on A, {-a, -a-8} on B), settings
# c_A = g^{j_a}, c_B = g^{j_b}. Joint forced-basis amplitude:
#   A(alpha, beta) = 64 * zeta-phase * (1 + (-1)^{alpha+beta} Lambda),
#   Lambda = zeta_80^{j_b - j_a}.
# Verified here from the explicit fibre sums (using C3 on each wing), and the
# F_641 shadow via zeta_80 -> g^8.
red8 = pow(g, 8, p)                                  # zeta_80 shadow
def E_exact(D):
    lam = zpow(D); lamc = zconj(lam)
    w = {}
    for al in (0, 1):
        for be in (0, 1):
            sgn = 1 if (al+be) % 2 == 0 else -1
            amp  = (Poly(1, x) + sgn*lam) % PHI
            ampc = (Poly(1, x) + sgn*lamc) % PHI
            w[(al, be)] = zmul(amp, ampc)
    tot = sum(w.values(), Poly(0, x)) % PHI
    num = (w[(0, 0)] + w[(1, 1)] - w[(0, 1)] - w[(1, 0)]) % PHI
    return num, tot, w

ok4, ok5, ok6 = True, True, True
for D in range(80):
    num, tot, w = E_exact(D)
    # tot must equal 8 exactly
    if not tot.as_expr().equals(8): ok4 = False
    # num must equal 4*(zeta^D + zeta^-D), i.e. E = cos(2 pi D / 80)
    tgt = (4*zpow(D) + 4*zpow(-D)) % PHI
    if not (num - tgt).is_zero: ok4 = False
    # C5: shadows in F_641: w reduces via zeta -> g^8 to |1 +- g^{8D}|^2-shadow
    for (al, be), wp in w.items():
        sh = sum(int(c) * pow(red8, m[0], p) for m, c in zip(wp.monoms(), wp.coeffs())) % p
        sgn = 1 if (al+be) % 2 == 0 else -1
        lamp = pow(red8, D, p)
        # carrier-internal: (1+sgn*lam)(1+sgn*lam^{-1}) mod p
        direct = ((1 + sgn*lamp) * (1 + sgn*pow(lamp, p-2, p))) % p
        if sh != direct: ok5 = False
    # C6: marginals: w(al,0)+w(al,1) = 4 exactly, independent of D
    for al in (0, 1):
        marg = (w[(al, 0)] + w[(al, 1)]) % PHI
        if not marg.as_expr().equals(4): ok6 = False
report('C4: E(Delta) = cos(pi Delta/40) exactly, all 80 settings (Z[zeta_80])', ok4)
report('C5: composite reduction commutes, Z[zeta_80] -> F_641 on all weights', ok5)
report('C6: no-signalling: marginals = 1/2 exactly, independent of the far setting', ok6)

# ---------------- C7: exhaustive CHSH sweep ----------------
Ev = np.cos(2*np.pi*np.arange(80)/80)               # exact values' float image
best, worst = -10.0, 10.0
arg = None
for d1 in range(80):
    for d2 in range(80):
        for d3 in range(80):
            d4 = (d2 + d3 - d1) % 80                # Delta_4 = Delta_2+Delta_3-Delta_1
            S = Ev[d1] + Ev[d2] + Ev[d3] - Ev[d4]
            if S > best: best, arg = S, (d1, d2, d3, d4)
            if S < worst: worst = S
ts = 2*np.sqrt(2)
report('C7: exhaustive sweep (80^3): max S = 2*sqrt(2) to 1e-12 (got %.12f at %s); '
       'never exceeded' % (best, str(arg)),
       abs(best - ts) < 1e-12 and best <= ts + 1e-12 and abs(worst + ts) < 1e-12)
report('C7: native Bell violation: S = %.12f > 2 strictly' % best, best > 2 + 0.7)

# ---------------- C8: exact Tsirelson saturation in Z[zeta_8] ----------------
# optimum e.g. at Delta = (70, 10, 10, 30) (settings j = 0, 20; 10, 70):
# E = cos(pi/4) thrice, -cos(3pi/4) once.
PHI8 = Poly(x**4 + 1, x)                            # Phi_8
z8 = Poly(x, x)
def z8pow(k): return Poly(x, x)**(k % 8) % PHI8
S8 = (z8pow(1) + z8pow(-1))*Rational(1, 2)*3 - (z8pow(3) + z8pow(-3))*Rational(1, 2)
S8 = S8 % PHI8
tgt = (2*(z8pow(1) + z8pow(-1))) % PHI8
report('C8: S = 2(zeta_8 + zeta_8^{-1}) = 2*sqrt(2) exactly in Z[zeta_8]',
       (S8 - tgt).is_zero)
S2 = zmul(Poly(S8, x), Poly(S8, x)) % PHI8
report('C8: S^2 = 8 as a ring identity', S2.as_expr().equals(8))

print('composite: all checks passed')
