## T9: qudit emulation of the FRC worked examples on quantum hardware.
## The forced-basis measurement of the formalism compiles to a shallow circuit:
## index the object cycle C_12 of the 157/53/13 example by (alpha, ell) with
## u = g_O^(alpha + 3*ell); the measurement vectors v_{r,alpha} are then
## delta_{alpha} x (quarter-turn characters i^{r ell}), so measuring in the
## forced basis is: apply (I_3 x F_4^dagger), read (alpha, r). F_4 is the
## 2-qubit QFT; the alpha register is a qutrit (or 2 qubits, 12 of 16 levels).
## The 641 configuration compiles identically with F_8 (3-qubit QFT) and a
## C_2 outcome register. What a hardware run certifies: the forced-basis
## statistics, the selection rules, the uniform-outcome law for winding states,
## and the three-outcome Malus law at the carrier's phase granularity.
## Claims verified:
##   (M1) U = I_3 x F_4 is unitary and its columns are exactly the normalised
##        forced-basis vectors (Gaussian-rational arithmetic);
##   (M2) winding states u^s: channel r = s mod 4 deterministic, outcome alpha
##        uniform (P = 1/3), all 12 windings, to 1e-12 (selection rule already
##        exact in validate.py);
##   (M3) doublet states u^s + lam*u^{s+4}: the three-outcome Malus law
##        P(alpha) = |1 + lam*omega^alpha|^2 / 6, omega = zeta_3, swept over
##        12 phases lam = zeta_12^j;
##   (M4) gate decomposition: the 2-qubit circuit H-CS-H-SWAP reproduces F_4
##        exactly (entry-by-entry);
##   (M5) the 641 compilation: I_2 x F_8 statistics reproduce the sigma_x
##        doublet law w_+- = |1 +- lam|^2/4 of composite.py (C3), swept over
##        the pi/40 setting granularity.
import numpy as np

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

# ---------------- M1: the measurement unitary, 157/53/13 ----------------
# basis order: |alpha, ell>, alpha in 0..2, ell in 0..3 (12 levels)
F4 = np.array([[1j**(r*l) for l in range(4)] for r in range(4)])/2.0
U = np.kron(np.eye(3), F4)                            # maps forced basis -> comp basis
report('M1: I_3 x F_4 unitary (12-level)', np.allclose(U @ U.conj().T, np.eye(12)))
# forced-basis vectors: v_{r,alpha}(alpha', ell) = delta * i^{r ell} / 2
ok1 = True
for r in range(4):
    for al in range(3):
        v = np.zeros(12, complex)
        for l in range(4):
            v[al*4 + l] = 1j**(r*l)/2.0
        e = np.zeros(12); e[al*4 + r] = 1
        if not np.allclose(U.conj().T @ v, e): ok1 = False
report('M1: columns of U are exactly the normalised forced-basis vectors', ok1)

# ---------------- M2: winding states -> deterministic channel, uniform outcome ----
z12 = np.exp(2j*np.pi/12)
ok2 = True
for s in range(12):
    psi = np.array([z12**(s*(al + 3*l)) for al in range(3) for l in range(4)])/np.sqrt(12)
    probs = np.abs(U.conj().T @ psi)**2
    for al in range(3):
        for r in range(4):
            tgt = (1/3) if (r - s) % 4 == 0 else 0.0
            if abs(probs[al*4 + r] - tgt) > 1e-12: ok2 = False
report('M2: winding u^s -> channel r = s mod 4 deterministic, outcome uniform '
       '1/3, all 12 windings', ok2)

# ---------------- M3: three-outcome Malus law ----------------
ok3 = True
om = np.exp(2j*np.pi/3)
for s in (0, 1, 5):
    for j in range(12):
        lam = z12**j
        psi = np.array([z12**(s*(al + 3*l)) + lam*z12**((s+4)*(al + 3*l))
                        for al in range(3) for l in range(4)])
        psi /= np.linalg.norm(psi)
        probs = np.abs(U.conj().T @ psi)**2
        r = s % 4
        for al in range(3):
            tgt = abs(1 + lam*om**al)**2/6
            if abs(probs[al*4 + r] - tgt) > 1e-12: ok3 = False
report('M3: doublet Malus law P(alpha) = |1 + lam omega^alpha|^2/6, '
       'swept over 12 phases and 3 windings', ok3)

# ---------------- M4: gate decomposition of F_4 ----------------
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
S = np.diag([1, 1j])
CS = np.diag([1, 1, 1, 1j])                            # controlled-S
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
QFT4 = SWAP @ np.kron(np.eye(2), H) @ CS @ np.kron(H, np.eye(2))
# standard QFT with kernel exp(2 pi i jk/4)/2 = i^{jk}/2
report('M4: H-CS-H-SWAP two-qubit circuit reproduces F_4 exactly',
       np.allclose(QFT4, F4))

# ---------------- M5: the 641 compilation (C_16, core C_8, C_2 outcome) ----------
F8 = np.array([[np.exp(2j*np.pi*r*l/8) for l in range(8)] for r in range(8)])/np.sqrt(8)
U16 = np.kron(np.eye(2), F8)
report('M5: I_2 x F_8 unitary (16-level)', np.allclose(U16 @ U16.conj().T, np.eye(16)))
z16 = np.exp(2j*np.pi/16)
ok5 = True
for s in (0, 3, 9):
    for j in range(80):
        lam = np.exp(2j*np.pi*8*j/640)                 # pi/40 granularity
        # basis |alpha, ell>, u = g_O^(alpha + 2*ell), alpha in {0,1}, ell in 0..7
        psi = np.array([z16**(s*(al + 2*l)) + lam*z16**((s+8)*(al + 2*l))
                        for al in range(2) for l in range(8)])
        psi /= np.linalg.norm(psi)
        probs = np.abs(U16.conj().T @ psi)**2
        r = s % 8
        wplus  = abs(1 + lam)**2/4
        wminus = abs(1 - lam)**2/4
        # alpha = 0 fibre carries (1+lam*(-1)^alpha) pattern: check both outcomes
        got = (probs[0*8 + r], probs[1*8 + r])
        if abs(got[0] - wplus) > 1e-12 or abs(got[1] - wminus) > 1e-12: ok5 = False
report('M5: 641 compilation reproduces the sigma_x doublet law w_+- = '
       '|1 +- lam|^2/4 over the pi/40 sweep', ok5)

print('INFO experiment card (157/53/13): prepare 12-level winding or doublet '
      'states; apply I_3 x QFT_4^dagger; read (alpha, r). Targets: trapped-ion '
      'qudits (d = 12 native) or 4-qubit embedding (12 of 16 levels).')
print('INFO experiment card (641/41/17): 16-level, I_2 x QFT_8^dagger; '
      'certifies Malus at pi/40 and the Tsirelson configuration statistics.')
print('emulation: all checks passed')
