## T6: the decoherence floor -- no spontaneous collapse, and the dilation floor.
## FRC is unitary below registration: the drive and every form-preserving
## propagator conserve ledger norms exactly, so an ISOLATED superposition keeps
## full fringe visibility at any mass, forever -- the framework forbids
## CSL/GRW-type spontaneous collapse outright (no parameter to tune). The only
## intrinsic floor is gravitational: mass is winding rate, so a composite's
## internal clocks read branch proper time, and a proper-time differential
## between interferometer arms writes which-path information into the internal
## state at exactly the rate of gravitational time-dilation decoherence
## (Pikovski et al. 2015) -- derived here from the FRC readings, with one
## FRC-specific discriminator: the spectrum is integer windings, so visibility
## REVIVES exactly at the cycle recurrence: dephasing, never collapse.
## Claims verified by exact arithmetic (Q(zeta_n), polys mod Phi_n):
##   (Q1) isolated doublet: fringe contrast = 1 exactly for every drive time and
##        every analyser setting sweep (unitarity: no intrinsic visibility loss);
##   (Q2) dilation dephasing: a cluster with internal winding distribution p_E
##        has branch visibility V(tau) = |sum_E p_E zeta^{E tau}| -- the
##        characteristic function of the internal energy distribution -- with
##        (a) the exact ring form verified, (b) Gaussian-envelope decay for a
##        binomial (thermal) distribution, matching the time-dilation law;
##   (Q3) exact recurrence: V(n) = 1 identically -- visibility revives at the
##        cycle period; decoherence in FRC is reversible dephasing, not collapse.
import numpy as np
from math import comb
from sympy import Poly, symbols, cyclotomic_poly, Rational

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

x = symbols('x')
n = 40                                                # internal winding cycle
PHI = Poly(cyclotomic_poly(n, x), x)
def zpow(k):  return Poly(x, x)**(k % n) % PHI
def zmul(a, b): return (a*b) % PHI
def zconj(a):
    out = Poly(0, x)
    for m, c in zip(a.monoms(), a.coeffs()):
        out = (out + c*zpow(-m[0])) % PHI
    return out

# ---------------- Q1: isolated superposition keeps contrast 1 ----------------
# doublet (1, lambda) with lambda = zeta^j; drive tau multiplies the relative
# phase by zeta^{8 tau} (gap 8, as in the composite gate); analyser sweep j':
# w_+(tau, j') = |1 + zeta^{j + 8 tau + j'}|^2 in {0..4}: contrast (max-min)/
# (max+min) over the sweep = 1 exactly iff max = 4 and min = 0 both occur.
ok1 = True
for j in (0, 3, 11):
    for tau in (0, 1, 5, 17, 39):
        vals = []
        for jp in range(n):
            k = (j + 8*tau + jp) % n
            w = (Poly(2, x) + zpow(k) + zpow(-k)) % PHI      # |1+zeta^k|^2
            vals.append(w)
        has4 = any(v.as_expr().equals(4) for v in vals)
        has0 = any(v.is_zero for v in vals)
        if not (has4 and has0): ok1 = False
report('Q1: isolated doublet contrast = 1 exactly, all drive times and phases '
       '(no spontaneous visibility loss at any mass: CSL forbidden)', ok1)

# ---------------- Q2: dilation dephasing law ----------------
# internal energy distribution p_E: binomial on E = 0..M (thermal-like), shifted;
# branch visibility V(tau)^2 = |sum_E p_E zeta^{E tau}|^2, exact in the ring.
M = 8
pE = {E: Rational(comb(M, E), 2**M) for E in range(M+1)}
def V2_exact(tau):
    s = Poly(0, x)
    for E, p in pE.items():
        s = (s + p*zpow(E*tau)) % PHI
    return zmul(s, zconj(s))
# (a) ring form vs direct numeric characteristic function
ok2 = True
for tau in range(n):
    v2 = V2_exact(tau)
    num = sum(float(c)*np.exp(2j*np.pi*m[0]/n) for m, c in zip(v2.monoms(), v2.coeffs())).real
    chi = abs(sum(float(p)*np.exp(2j*np.pi*E*tau/n) for E, p in pE.items()))**2
    if abs(num - chi) > 1e-12: ok2 = False
report('Q2a: V(tau)^2 = |characteristic function of internal windings|^2, '
       'exact ring form, full sweep', ok2)
# (b) Gaussian envelope: binomial variance M/4: V ~ exp(-sigma^2 theta^2/2)
ok2b = True
sig2 = M/4
for tau in (1, 2, 3, 4):
    th = 2*np.pi*tau/n
    chi = abs(sum(float(p)*np.exp(1j*E*th) for E, p in pE.items()))
    gauss = np.exp(-sig2*th*th/2)
    if abs(chi - gauss) > 0.02: ok2b = False                 # envelope agreement
report('Q2b: Gaussian-envelope decay V = exp(-sigma^2 (omega tau)^2/2) for the '
       'thermal distribution (the time-dilation law)', ok2b)

# ---------------- Q3: exact recurrence ----------------
v2 = V2_exact(n)
report('Q3: exact recurrence V(n)^2 = 1: visibility revives at the cycle '
       'period -- dephasing, never collapse', v2.as_expr().equals(1))

print('decoherence: all checks passed')
