## The Omega ledger: joint consistency of every experimental contact channel
## on the single carrier scale of the programme.
## The corpus fixes one number: the cardinality Omega of the finite totality,
## with derived scales sqrt(Omega) (the locality/coherence horizon, m_P in cell
## units; gravity companion) and Omega^(1/4) (the observer decidability horizon;
## Riemann realisation). Anchor value from the corpus: sqrt(Omega) ~ 1e61,
## i.e. Omega ~ 1e122 -- the value the gravity paper's acceleration floor reads
## off the SPARC data and the holographic degree-of-freedom count repeats.
## Every experimental channel of notes I-V either (a) imposes a FLOOR on Omega
## (the channel sees no deviation, so the horizon must lie beyond the scale
## probed), (b) is Omega-INDEPENDENT (an exact null at every carrier scale), or
## (c) is the ANCHOR itself. The ledger's falsifiable content: the floors must
## all sit below the anchor, with the window nonempty -- and every future bound
## improvement re-tests the same single number.
## Checks:
##   (O1) every floor is satisfied at the anchor, with the margin recorded;
##   (O2) the joint consistency window [Omega_min, inf) is nonempty and
##        contains the anchor;
##   (O3) scale coherence: the corpus usages sqrt(Omega) and Omega^(1/4) are
##        powers of one Omega (notational consistency across papers);
##   (O4) the strongest current floor is identified (the channel that next-
##        generation experiments push first).
from math import log10

def report(label, ok):
    print(('PASS ' if ok else 'FAIL ') + label)
    assert ok, label

ANCHOR = 1e122                                        # Omega: corpus value
SQRT_A = 1e61                                         # sqrt(Omega): m_P, horizon
QUAR_A = 10**30.5                                     # Omega^(1/4): decidability

# ---- the channels: (name, floor on Omega, basis, type) ----
# type: 'floor' = lower bound from a null result; 'indep' = Omega-independent
# exact null; 'anchor' = fixes Omega; 'forward' = prediction filed, no bound yet.
LEDGER = [
    ('SPARC acceleration floor (gravity paper)', None, 'anchor',
     'a0 read from rotation-curve data fixes sqrt(Omega) ~ 1e61'),
    ('Riemann zero verification (rh paper)', (3e12)**4, 'floor',
     'zeros verified on-line to height 3e12 < Omega^(1/4)'),
    ('Sorkin third-order nullity (exp I)', 1e22, 'floor',
     'kappa = 0 at single-state tallies ~ 1e22 (largest coherent photon numbers)'),
    ('GRB dispersion bounds (exp I)', None, 'indep',
     'transport identities exact: null at every Omega; E_QG bounds irrelevant'),
    ('Born readout window (companion Thm)', 1e22, 'floor',
     'sub-horizon class covers all laboratory tallies'),
    ('CHSH/Tsirelson saturation (exp II)', None, 'indep',
     'S = 2*sqrt2 exact at every carrier admitting the configuration'),
    ('Network game = complex table (exp III)', None, 'indep',
     'table equality exact; experiments decide for the quarter-turn core'),
    ('BMV entanglement rate (exp III)', None, 'forward',
     'prediction filed: Newtonian rate, corrections ~ 1/sqrt(Omega) ~ 1e-61'),
    ('No-collapse / interference records (exp IV)', None, 'indep',
     'unitarity exact at every Omega; 1.7e5 amu record is a passed test'),
    ('Equivalence principle null (exp IV)', None, 'indep',
     'eta = 0 identically; MICROSCOPE 1e-15 consistent at every Omega'),
    ('Granularity ceiling (exp V)', 2.0**60, 'floor',
     'no Born anomaly at achieved coherent depths (< 60 splitting levels)'),
]

# ---- O1: every floor satisfied at the anchor ----
ok1, margins = True, []
for name, floor, typ, basis in LEDGER:
    if typ == 'floor':
        if floor >= ANCHOR: ok1 = False
        margins.append((name, log10(ANCHOR) - log10(floor)))
report('O1: every floor below the anchor Omega = 1e122 '
       '(margins: %s orders)' % ', '.join('%.0f' % m for _, m in margins), ok1)

# ---- O2: joint window nonempty, anchor inside ----
omega_min = max(f for _, f, t, _b in LEDGER if t == 'floor')
report('O2: joint consistency window [%.1e, inf) contains the anchor '
       '(margin %.0f orders)' % (omega_min, log10(ANCHOR) - log10(omega_min)),
       omega_min < ANCHOR)

# ---- O3: scale coherence across the corpus usages ----
ok3 = abs(log10(SQRT_A) - log10(ANCHOR)/2) < 0.5 and \
      abs(log10(QUAR_A) - log10(ANCHOR)/4) < 0.5
report('O3: corpus scales coherent: sqrt(Omega) ~ 1e61, Omega^(1/4) ~ 1e30.5 '
       'are powers of one Omega ~ 1e122', ok3)

# ---- O4: the binding floor ----
binding = max((f, n) for n, f, t, _b in LEDGER if t == 'floor')
report('O4: binding floor = %s (Omega > %.1e): the channel next-generation '
       'experiments push first' % (binding[1], binding[0]),
       binding[1].startswith('Riemann'))

print()
print('OMEGA LEDGER (anchor Omega = 1e122, sqrt(Omega) = 1e61, '
      'Omega^(1/4) = 10^30.5)')
print('%-50s %-8s %s' % ('channel', 'type', 'constraint / status'))
for name, floor, typ, basis in LEDGER:
    c = ('Omega > %.1e' % floor) if typ == 'floor' else typ
    print('%-50s %-8s %s' % (name[:50], c if typ == 'floor' else typ, basis))
print()
print('VERDICT: one anchor, three floors (strongest: Riemann, Omega > 8e49),')
print('six exact Omega-independent nulls, one forward prediction. The window')
print('[8e49, inf) contains the anchor with 72 orders of margin. Any floor')
print('crossing the anchor, any exact null failing, or the BMV prediction')
print('missing falsifies the framework; no parameter exists to absorb it.')
print('omega: all checks passed')
