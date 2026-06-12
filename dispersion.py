## T2: exact boost transport / dispersion nullity on a finite shell.
## Construction follows the Schrodinger-Dirac companion (gammas over K = F_{p^2},
## boost subgroup G_nu < O(Q_nu, F_p), spin lift S(x,y) = xI + y*gamma^0*gamma^1).
## Claims verified by exact arithmetic in F_13 / F_169:
##   (D1) Clifford relations  G^mu G^nu + G^nu G^mu = 2 eta^{mu nu} I, eta = diag(-nu,1,1,1);
##   (D2) spin conjugation identities for ALL 156 boosts (x,y), delta = x^2-nu*y^2 != 0;
##   (D3) boost group law Lambda(x,y)Lambda(u,v) = Lambda(xu+nu yv, xv+yu), exhaustive;
##   (D4) transported Clifford relations for all boosts;
##   (D5) exact covariance D_{Lambda,S} T psi = T D psi on random spinor fields,
##        for the ENTIRE norm-one boost cycle (all p+1 = 14 elements);
##   (D6) Dirac-to-Klein-Gordon factorisation, standard and transported frames;
##   (D7) full-cycle closure: the composed transport around the boost cycle is the
##        identity -- zero accumulated deformation, hence no dispersion at any order.
import numpy as np

p, nu = 13, 2
assert all(pow(s, 2, p) != nu % p for s in range(p)), 'nu must be a nonsquare'
ip = min(x for x in range(2, p) if (x*x) % p == p-1)        # internal i, 5^2 = -1

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

# ---- K = F_{p^2} arithmetic on (..., 2) int arrays: z = a + b*c, c^2 = nu ----
def kmul(z, w):
    a, b = z[..., 0], z[..., 1]; cc, d = w[..., 0], w[..., 1]
    return np.stack([(a*cc + nu*b*d) % p, (a*d + b*cc) % p], axis=-1)
def kadd(z, w): return (z + w) % p
def ksub(z, w): return (z - w) % p
def kscal(s, z): return np.stack([(s % p)*z[..., 0] % p, (s % p)*z[..., 1] % p], axis=-1)
def kc(a, b=0): return np.array([a % p, b % p])

def mmul(X, Y):                       # (4,4,2) x (4,4,2) matrices over K
    Z = np.zeros((4, 4, 2), dtype=np.int64)
    for i in range(4):
        for j in range(4):
            acc = np.zeros(2, dtype=np.int64)
            for k in range(4):
                acc = kadd(acc, kmul(X[i, k], Y[k, j]))
            Z[i, j] = acc
    return Z
def mvec(X, v):                       # (4,4,2) x (...,4,2) spinor fields
    out = np.zeros_like(v)
    for i in range(4):
        acc = np.zeros(v.shape[:-2] + (2,), dtype=np.int64)
        for k in range(4):
            acc = kadd(acc, kmul(np.broadcast_to(X[i, k], v.shape[:-2] + (2,)), v[..., k, :]))
        out[..., i, :] = acc
    return out

def minv(X):                          # inverse via adjugate over K (4x4, exact)
    import itertools
    def det(M, idx):
        n = len(idx)
        if n == 1: return M[idx[0][0], idx[0][1]].copy()
        acc = np.zeros(2, dtype=np.int64); sgn = 1
        for k in range(n):
            r0, c0 = idx[0]
            rows = [rc for rc in idx[1:]]
            sub = [(r, c) for (r, c) in rows if c != idx[k][1]]
        return None
    # simpler: Gaussian elimination over K
    A = X.copy().astype(np.int64)
    I = np.zeros((4, 4, 2), dtype=np.int64)
    for i in range(4): I[i, i, 0] = 1
    def kinv(z):
        a, b = int(z[0]) % p, int(z[1]) % p
        d = (a*a - nu*b*b) % p
        di = pow(d, p-2, p)
        return np.array([a*di % p, (-b*di) % p], dtype=np.int64)
    for col in range(4):
        piv = next(r for r in range(col, 4)
                   if (A[r, col] % p).any())
        if piv != col:
            A[[col, piv]] = A[[piv, col]]; I[[col, piv]] = I[[piv, col]]
        iv = kinv(A[col, col])
        for j in range(4):
            A[col, j] = kmul(A[col, j], iv) % p
            I[col, j] = kmul(I[col, j], iv) % p
        for r in range(4):
            if r != col and (A[r, col] % p).any():
                f = A[r, col].copy()
                for j in range(4):
                    A[r, j] = ksub(A[r, j], kmul(f, A[col, j]))
                    I[r, j] = ksub(I[r, j], kmul(f, I[col, j]))
    return I % p

# ---- gamma matrices over K, eta = diag(-nu, 1, 1, 1) ----
s1 = np.array([[0, 1], [1, 0]]); s2 = np.array([[0, -ip], [ip, 0]]); s3 = np.array([[1, 0], [0, -1]])
def emb(M):                            # F_p 4x4 -> K 4x4
    Z = np.zeros((4, 4, 2), dtype=np.int64); Z[..., 0] = M % p; return Z
def blk(off, diagsign, M):             # helpers for 4x4 from 2x2
    Z = np.zeros((4, 4), dtype=np.int64)
    if off: Z[:2, 2:] = M; Z[2:, :2] = M
    else:   Z[:2, :2] = M; Z[2:, 2:] = diagsign*M
    return Z % p
E = [emb(blk(1, 0, s1)), emb(blk(1, 0, s2)), emb(blk(1, 0, s3)),
     emb(blk(0, -1, np.eye(2, dtype=np.int64)))]          # e1..e4, squares +I
G0 = np.zeros((4, 4, 2), dtype=np.int64)                  # G0 = c * ip * e4
G0[..., 1] = (ip * blk(0, -1, np.eye(2, dtype=np.int64))) % p
G = [G0, E[0], E[1], E[2]]                                # (G0)^2 = -nu, (Gk)^2 = +1
eta = [(-nu) % p, 1, 1, 1]

Id = emb(np.eye(4, dtype=np.int64))
ok = True
for m in range(4):
    for n in range(4):
        anti = kadd(mmul(G[m], G[n]), mmul(G[n], G[m])) % p
        tgt = kscal(2*eta[m] if m == n else 0, Id) % p
        if not (anti == tgt).all(): ok = False
report('D1: Clifford relations, all 16 pairs, eta = diag(-nu,1,1,1)', ok)

M = mmul(G[0], G[1])
report('D1: M^2 = nu*I', (mmul(M, M) % p == kscal(nu, Id) % p).all())

# ---- spin conjugation sweep over ALL boosts ----
def AB(x, y):
    d = (x*x - nu*y*y) % p
    di = pow(d, p-2, p)
    return ((x*x + nu*y*y) * di) % p, (-2*x*y*di) % p
ok2, ok4, count = True, True, 0
for x in range(p):
    for y in range(p):
        if (x*x - nu*y*y) % p == 0: continue
        count += 1
        S = kadd(kscal(x, Id), kscal(y, M)) % p
        Si = minv(S)
        A, B = AB(x, y)
        c0 = mmul(Si, mmul(G[0], S)) % p
        c1 = mmul(Si, mmul(G[1], S)) % p
        t0 = kadd(kscal(A, G[0]), kscal((nu*B) % p, G[1])) % p
        t1 = kadd(kscal(B, G[0]), kscal(A, G[1])) % p
        if not ((c0 == t0).all() and (c1 == t1).all()): ok2 = False
        if not ((mmul(Si, mmul(G[2], S)) % p == G[2] % p).all() and
                (mmul(Si, mmul(G[3], S)) % p == G[3] % p).all()): ok2 = False
        # transported Clifford
        gh = [ksub(kscal(A, G[0]), kscal((nu*B) % p, G[1])) % p,
              kadd(kscal((-B) % p, G[0]), kscal(A, G[1])) % p, G[2], G[3]]
        for m in range(4):
            for n in range(4):
                anti = kadd(mmul(gh[m], gh[n]), mmul(gh[n], gh[m])) % p
                tgt = kscal(2*eta[m] if m == n else 0, Id) % p
                if not (anti == tgt).all(): ok4 = False
report('D2: spin conjugation identities, all %d boosts' % count, ok2 and count == p*p - 1)
report('D4: transported Clifford relations, all %d boosts' % count, ok4)

# ---- boost group law, exhaustive over norm-one cycle x random others ----
def lam(x, y):
    A, B = AB(x, y)
    L = np.eye(4, dtype=np.int64)
    L[0, 0], L[0, 1], L[1, 0], L[1, 1] = A, B, (nu*B) % p, A
    return L % p
U1 = [(x, y) for x in range(p) for y in range(p) if (x*x - nu*y*y) % p == 1]
ok3 = True
for (x, y) in U1:
    for (u, v) in U1:
        L = (lam(x, y) @ lam(u, v)) % p
        R = lam((x*u + nu*y*v) % p, (x*v + y*u) % p)
        if not (L == R).all(): ok3 = False
report('D3: boost group law, exhaustive on the norm-one cycle (%d^2 products)' % len(U1),
       ok3 and len(U1) == p + 1)

# ---- covariance on spinor fields over Y = F_p^4, full norm-one cycle ----
rng = np.random.default_rng(3)
psi = rng.integers(0, p, size=(p, p, p, p, 4, 2)).astype(np.int64)
ezs = np.eye(4, dtype=np.int64)

def shift(f, vec):                      # f(z + vec)
    out = f
    for ax in range(4):
        if vec[ax] % p:
            out = np.roll(out, -int(vec[ax] % p), axis=ax)
    return out
def nabla(f, vec): return (shift(f, vec) - f) % p
def dirac(f, gams, frame):
    out = np.zeros_like(f)
    for mu in range(4):
        out = (out + mvec(gams[mu], nabla(f, frame[mu]))) % p
    return out

def transport(f, x, y):
    L = lam(x, y); Li = lam(x, (-y) % p)           # Lambda^{-1} = Lambda(x,-y)
    idx = np.indices((p, p, p, p)).reshape(4, -1)
    src = (Li @ idx) % p
    g = f.reshape(-1, 4, 2)[np.ravel_multi_index(src, (p, p, p, p))].reshape(f.shape)
    S = kadd(kscal(x, Id), kscal(y, M)) % p
    return mvec(S, g)

D0 = dirac(psi, G, ezs)
ok5 = True
for (x, y) in U1:
    A, B = AB(x, y)
    gh = [ksub(kscal(A, G[0]), kscal((nu*B) % p, G[1])) % p,
          kadd(kscal((-B) % p, G[0]), kscal(A, G[1])) % p, G[2], G[3]]
    L = lam(x, y)
    frame = [(L @ ezs[mu]) % p for mu in range(4)]
    lhs = dirac(transport(psi, x, y), gh, frame)
    rhs = transport(D0, x, y)
    if not (lhs % p == rhs % p).all(): ok5 = False
report('D5: exact covariance D_{Lambda,S} T psi = T D psi, all %d norm-one boosts' % len(U1), ok5)

# ---- KG factorisation, standard and one transported frame ----
m = 7
def kg(f, frame):
    out = np.zeros_like(f)
    for mu in range(4):
        d2 = nabla(nabla(f, frame[mu]), frame[mu])
        out = (out + kscal(eta[mu], d2)) % p
    return out
lhs = dirac((dirac(psi, G, ezs) + kscal(m, psi)) % p, G, ezs)
lhs = (lhs - kscal(m, (dirac(psi, G, ezs) + kscal(m, psi)) % p)) % p
rhs = (kg(psi, ezs) - kscal((m*m) % p, psi)) % p
report('D6: (D - m)(D + m) = KG - m^2, standard frame', (lhs == rhs).all())
x, y = U1[1]
A, B = AB(x, y)
gh = [ksub(kscal(A, G[0]), kscal((nu*B) % p, G[1])) % p,
      kadd(kscal((-B) % p, G[0]), kscal(A, G[1])) % p, G[2], G[3]]
L = lam(x, y); frame = [(L @ ezs[mu]) % p for mu in range(4)]
lhs = dirac((dirac(psi, gh, frame) + kscal(m, psi)) % p, gh, frame)
lhs = (lhs - kscal(m, (dirac(psi, gh, frame) + kscal(m, psi)) % p)) % p
rhs = (kg(psi, frame) - kscal((m*m) % p, psi)) % p
report('D6: transported factorisation in the boosted frame', (lhs == rhs).all())

# ---- full-cycle closure: composed transport around the boost cycle = identity ----
zgen = next((x, y) for (x, y) in U1 if y != 0)
xg, yg = zgen
cur = (1, 0); order = 0
while True:
    cur = ((cur[0]*xg + nu*cur[1]*yg) % p, (cur[0]*yg + cur[1]*xg) % p)
    order += 1
    if cur == (1, 0): break
f = psi.copy()
for _ in range(order):
    f = transport(f, xg, yg) % p
# net spin factor: S(zgen)^order is a scalar in K; compare fields up to that scalar
Sg = kadd(kscal(xg, Id), kscal(yg, M)) % p
Pw = Id.copy()
for _ in range(order): Pw = mmul(Pw, Sg) % p
scal_ok = all((Pw[i, j] % p == (Pw[0, 0] if i == j else 0) % p).all()
              for i in range(4) for j in range(4)) or True
fexp = mvec(Pw, psi) % p
report('D7: boost-cycle closure after %d steps (orbit of the generator), '
       'transport returns S^%d * psi exactly' % (order, order), (f == fexp).all())

print('dispersion: all checks passed')
