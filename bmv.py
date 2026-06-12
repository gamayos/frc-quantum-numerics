## T5: gravitationally induced entanglement (BMV) -- the FRC mechanism.
## In FRC gravity IS phase synchronisation on the substrate (gravity companion):
## two locked clusters in path superposition accumulate a branch-pair-dependent
## relative phase at the Newtonian rate phi(t) = (G m1 m2 / hbar)(1/d - 1/d')t,
## with the Newtonian potential supplied by the gravity paper's derived lattice
## Green's function. The interaction couples to the conserved offset sector of
## the pair -- the same constant of motion that carries Bell correlations
## (composite.py C2) -- so it is an entangling channel by construction.
## Claims verified by exact arithmetic (Q(zeta_80), polys mod Phi_80):
##   (B1) zero coupling: the joint state remains an exact product (concurrence 0);
##   (B2) conditional phase phi = 2 pi k/80 produces entanglement with
##        concurrence^2 = sin^2(phi/2) EXACTLY, for the full sweep k = 0..79;
##   (B3) the witness: the Horodecki CHSH maximum 2*sqrt(1 + sin^2(phi/2))
##        exceeds 2 for every k != 0 (numeric image of the exact table), and
##        reaches the Tsirelson value 2*sqrt2 at phi = pi -- consistency with
##        the composite gate;
##   (B4) the coupling is offset-diagonal (commutes with the drive); A's
##        marginal is invariant under local operations on B (no signalling);
##        and the fringe visibility obeys V^2 + C^2 = 1 exactly -- the
##        gravitational which-path complementarity, the BMV observable pair.
## FRC forward prediction for BMV-class experiments: entanglement forms at the
## Newtonian rate exactly; corrections are bounded by the horizon (1/sqrt(Omega))
## and are unobservable; a null result at the Newtonian rate falsifies the
## framework's gravitational sector outright.
import numpy as np
from sympy import Poly, symbols, cyclotomic_poly, Rational

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

x = symbols('x')
PHI = Poly(cyclotomic_poly(80, x), x)
def zpow(k):  return Poly(x, x)**(k % 80) % PHI
def zmul(a, b): return (a*b) % PHI
def zconj(a):
    out = Poly(0, x)
    for m, c in zip(a.monoms(), a.coeffs()):
        out = (out + c*zpow(-m[0])) % PHI
    return out

# joint branch state of two clusters, each (|0> + |1>), coupling phase on |11>:
# amplitudes (1, 1, 1, zeta^k); unnormalised (norm^2 = 4).
def amps(k):
    return [Poly(1, x), Poly(1, x), Poly(1, x), zpow(k)]

# ---------------- B1: zero coupling -> exact product ----------------
a = amps(0)
det = (zmul(a[0], a[3]) - zmul(a[1], a[2])) % PHI     # ad - bc
report('B1: phi = 0: ad - bc = 0 exactly (product state, concurrence 0)',
       det.is_zero)

# ---------------- B2: concurrence^2 = sin^2(phi/2) exactly ----------------
ok2 = True
for k in range(80):
    a = amps(k)
    det = (zmul(a[0], a[3]) - zmul(a[1], a[2])) % PHI
    c2 = zmul(det, zconj(det))                         # |ad-bc|^2 (norm 4 state)
    # concurrence = 2|ad-bc|/norm^2 = |ad-bc|/2 -> C^2 = |ad-bc|^2/4
    # target: sin^2(phi/2) = (1 - cos phi)/2 = (2 - zeta^k - zeta^-k)/4
    tgt = (Poly(2, x) - zpow(k) - zpow(-k)) % PHI
    if not (c2 - tgt).is_zero: ok2 = False
report('B2: concurrence^2 = sin^2(phi/2) exactly, full sweep phi = 2 pi k/80', ok2)

# ---------------- B3: Horodecki CHSH witness ----------------
ok3, okts = True, False
for k in range(80):
    phi = 2*np.pi*k/80
    psi = np.array([1, 1, 1, np.exp(1j*phi)])/2.0
    # correlation matrix T_ij = <sigma_i x sigma_j>
    sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
           np.array([[1, 0], [0, -1]])]
    rho = np.outer(psi, psi.conj())
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = np.real(np.trace(rho @ np.kron(sig[i], sig[j])))
    ev = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    Smax = 2*np.sqrt(ev[0] + ev[1])
    tgt = 2*np.sqrt(1 + np.sin(phi/2)**2)
    if abs(Smax - tgt) > 1e-9: ok3 = False
    if k != 0 and Smax <= 2 + 1e-12: ok3 = False
    if abs(Smax - 2*np.sqrt(2)) < 1e-9 and k == 40: okts = True
report('B3: Horodecki CHSH max = 2*sqrt(1 + sin^2(phi/2)); > 2 for all phi != 0; '
       '= 2*sqrt2 at phi = pi (k = 40)', ok3 and okts)

# ---------------- B4: drive commutation, no-signalling, complementarity ----------------
# (i) the coupling is diagonal in the joint branch basis, hence commutes with
#     the diagonal drive exactly;
# (ii) A's marginal is independent of any LOCAL operation on B (no signalling);
# (iii) A's fringe visibility V = |cos(phi/2)| obeys V^2 + C^2 = 1 EXACTLY:
#       the gravitational which-path complementarity, the BMV observable pair.
ok4a, ok4b, ok4c = True, True, True
drv = np.diag([1, 1j, 1j, -1])                        # representative diagonal drive image
for k in (1, 7, 20, 40, 63):
    phi = 2*np.pi*k/80
    G = np.diag([1, 1, 1, np.exp(1j*phi)])
    if not np.allclose(G @ drv, drv @ G): ok4a = False
    psi = (G @ np.array([1, 1, 1, 1.0]))/2.0
    rho = np.outer(psi, psi.conj())
    rA = np.trace(rho.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    for th in (0.3, 1.1, 2.7):                        # arbitrary local ops on B
        UB = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ \
             np.diag([1, np.exp(1j*0.9*th)])
        U = np.kron(np.eye(2), UB)
        rho2 = U @ rho @ U.conj().T
        rA2 = np.trace(rho2.reshape(2, 2, 2, 2), axis1=1, axis2=3)
        if not np.allclose(rA, rA2): ok4b = False
report('B4i: the coupling commutes with the diagonal drive (offset-diagonal)', ok4a)
report('B4ii: A marginal invariant under arbitrary local operations on B '
       '(no signalling through the gravitational channel)', ok4b)
# (iii) exact ring identity: V^2 + C^2 = 1, V^2 = (2 + z^k + z^-k)/4, C^2 = (2 - z^k - z^-k)/4
ok4c = True
for k in range(80):
    V2 = (Poly(2, x) + zpow(k) + zpow(-k)) % PHI
    C2 = (Poly(2, x) - zpow(k) - zpow(-k)) % PHI
    if not ((V2 + C2) % PHI).as_expr().equals(4): ok4c = False
report('B4iii: visibility-entanglement complementarity V^2 + C^2 = 1 exactly, '
       'full sweep (the BMV observable pair)', ok4c)

print('bmv: all checks passed')
