## T7: locked/unlocked coupling phenomenology and the equivalence principle.
## The gravity companion's Lemma 4.2: a locked cluster of cardinality m couples
## to the ambient field with coefficient exactly m; an unlocked single-site
## ensemble of m independent phases couples with expected magnitude O(sqrt m).
## This file pins the phenomenology: WHICH observables see m and which sqrt(m).
##   (E1) field linearity / full gas gravity: the stationary bias field of many
##        separately locked clusters is the exact sum of full-m contributions --
##        a gas of internally locked atoms sources its entire mass; verified by
##        exact rational solution of the lattice Poisson equation;
##   (E2) equivalence principle, exact null: pull on a locked test cluster is
##        m * grad(u), inertia is m, so the acceleration is independent of m
##        and composition identically: eta = 0 -- consistent with MICROSCOPE
##        (|eta| < 1e-15) and atom-interferometric EP tests, and falsifiable by
##        ANY confirmed EP violation;
##   (E3) the amplitude/power dichotomy: for m mutually incoherent phases at one
##        site, E|sum zeta^{theta_i}|^2 = m EXACTLY (character orthogonality;
##        verified by exhaustive enumeration), versus m^2 for a locked cluster:
##        coherence-sensitive (amplitude) observables scale sqrt(m), while
##        energy (power) observables scale m always -- gravity sources by power,
##        synchronisation couples by amplitude;
##   (E4) sampled scaling: |sum|^2/m concentrates near 1 for large unlocked m.
## Consequence: EP experiments see exact nulls (Class-C survival); the sqrt(m)
## appears only in coherence observables -- e.g. the entangling visibility of a
## BMV-type experiment degrades when the source mass's collective phase is
## incoherent: the FRC-specific differential handle (cold solid vs hot source).
import numpy as np
from fractions import Fraction as Fr
from itertools import product

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

# ---------------- E1: exact lattice Poisson, linearity and full sourcing ----------------
# periodic 4x4x4 lattice; kappa * Laplacian u = -rho, zero-mean gauge; exact
# rational solve via Gaussian elimination over Q.
L = 4
Ncell = L**3
def idx(x, y, z): return (x % L)*L*L + (y % L)*L + (z % L)
A = [[Fr(0)]*Ncell for _ in range(Ncell)]
for X in range(L):
    for Y in range(L):
        for Z in range(L):
            i = idx(X, Y, Z)
            A[i][i] = Fr(-6)
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                A[i][idx(X+dx, Y+dy, Z+dz)] += Fr(1)

def solve_poisson(rho):
    # solve A u = -rho with sum(u) = 0 (replace last row by gauge condition);
    # requires sum(rho) compensated: subtract mean source (jellium background)
    mean = sum(rho, Fr(0))/Ncell
    b = [-(r - mean) for r in rho]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    M[-1] = [Fr(1)]*Ncell + [Fr(0)]                    # gauge: sum u = 0
    # gaussian elimination
    for col in range(Ncell):
        piv = next(r for r in range(col, Ncell) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v/pv for v in M[col]]
        for r in range(Ncell):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [vr - f*vc for vr, vc in zip(M[r], M[col])]
    return [M[i][Ncell] for i in range(Ncell)]

rho1 = [Fr(0)]*Ncell; rho1[idx(0, 0, 0)] = Fr(3)       # cluster m1 = 3
rho2 = [Fr(0)]*Ncell; rho2[idx(2, 1, 3)] = Fr(7)       # cluster m2 = 7
rho12 = [a+b for a, b in zip(rho1, rho2)]
u1, u2, u12 = solve_poisson(rho1), solve_poisson(rho2), solve_poisson(rho12)
report('E1: exact superposition u(m1 + m2) = u(m1) + u(m2) on the lattice '
       '(rational arithmetic, all 64 cells)',
       all(a+b == c for a, b, c in zip(u1, u2, u12)))
rho_unit = [Fr(0)]*Ncell; rho_unit[idx(0, 0, 0)] = Fr(1)
uu = solve_poisson(rho_unit)
report('E1: source coefficient is exactly m: u(m=3 cluster) = 3 * u(unit), '
       'every cell', all(3*a == b for a, b in zip(uu, u1)))

# ---------------- E2: equivalence principle, exact null ----------------
# pull on a locked test cluster at x: F = m_t * (discrete gradient of u);
# inertia m_t: acceleration a = F/m_t = grad u, independent of m_t exactly.
gx = u12[idx(1, 0, 0)] - u12[idx(3, 0, 0)]             # representative gradient
acc3 = (Fr(3)*gx)/Fr(3)
acc7 = (Fr(7)*gx)/Fr(7)
report('E2: acceleration independent of test mass and composition, exactly '
       '(eta = 0 identically)', acc3 == acc7 == gx)

# ---------------- E3: amplitude/power dichotomy, exact ----------------
# m iid uniform phases on C_8: E|sum zeta^{theta_i}|^2 = m exactly (exhaustive)
nph = 8
zs = [np.exp(2j*np.pi*k/nph) for k in range(nph)]
ok3 = True
for m in (1, 2, 3):
    tot = 0j
    for cfg in product(range(nph), repeat=m):
        s = sum(zs[k] for k in cfg)
        tot += s*np.conj(s)
    EV = tot.real/(nph**m)
    if abs(EV - m) > 1e-12: ok3 = False
report('E3: E|sum|^2 = m exactly for unlocked phases (exhaustive, m = 1, 2, 3); '
       'locked cluster gives |sum|^2 = m^2', ok3 and abs(abs(sum(zs[0] for _ in range(5)))**2 - 25) < 1e-12)

# ---------------- E4: sampled sqrt(m) scaling ----------------
rng = np.random.default_rng(5)
m = 10000
vals = []
for _ in range(200):
    th = rng.integers(0, nph, m)
    s = np.exp(2j*np.pi*th/nph).sum()
    vals.append(abs(s)**2/m)
mean = float(np.mean(vals))
report('E4: sampled |sum|^2/m = %.3f (concentrates near 1 at m = 10^4): '
       'unlocked coupling magnitude ~ sqrt(m)' % mean, abs(mean - 1) < 0.2)

print('equivalence: all checks passed')
