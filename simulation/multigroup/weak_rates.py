"""Energy-dependent neutrino interaction rates Gamma(E) for the processes of Hannestad & Madsen (1995), Tables I-II,
for massless particles in kinetic equilibrium at a common temperature T (units T = 1), with Fermi-Dirac statistics
(target occupation and Pauli blocking of both final states) or Boltzmann statistics (no blocking).
S|M|^2 = 32 G_F^2 [A (p1.p2)^2 + B (p1.p4)^2 + C (p1.p3)^2]; particle 1 is the neutrino whose rate is computed.
Gamma(E1) = int d^3p2/(2pi)^3 f2 (1/(4 E1 E2)) int dOmega*/(32 pi^2) S|M|^2 (1-f3)(1-f4).
Output: Gamma/(G_F^2 T^4 E) for nu-e (plasma) and nu-nu processes, for nu_e and nu_mu,tau."""
import numpy as np, json, os
rng = np.random.default_rng(12345)
s2w = 0.231
CV, CA = 0.5 + 2*s2w, 0.5
fd = lambda E: 1.0/(np.exp(np.clip(E, -50, 700)) + 1.0)

# (A, B, C) coefficients and whether final states are blocked; target = 'e' (both spins summed in |M|^2) or 'nu'
def processes(flavour):
    if flavour == 'e':
        L, R = (CA + CV)**2, (CA - CV)**2
        nu_nu = [((0, 4, 0), 'nu'),          # nu_e nubar_e -> nu_e nubar_e
                 ((1, 0, 0), 'nu'),          # nu_e nu_e -> nu_e nu_e (S included)
                 ((1, 0, 0), 'nu'), ((1, 0, 0), 'nu'),      # nu_e nu_mu, nu_e nu_tau
                 ((0, 1, 0), 'nu'), ((0, 1, 0), 'nu'),      # nu_e nubar_mu, nu_e nubar_tau
                 ((0, 1, 0), 'nu'), ((0, 1, 0), 'nu')]      # nu_e nubar_e -> nu_mu nubar_mu, -> nu_tau nubar_tau
    else:
        L, R = (CV + CA - 2)**2, (CA - CV)**2
        nu_nu = [((0, 4, 0), 'nu'),          # nu_i nubar_i -> nu_i nubar_i
                 ((0, 1, 0), 'nu'),          # nu_i nubar_j (j = other non-e flavour)
                 ((1, 0, 0), 'nu'), ((1, 0, 0), 'nu'),      # nu_i nu_i, nu_i nu_j
                 ((0, 1, 0), 'nu'), ((1, 0, 0), 'nu'),      # nu_i nubar_e, nu_i nu_e
                 ((0, 1, 0), 'nu'), ((0, 1, 0), 'nu')]      # nu_i nubar_i -> nu_e nubar_e, -> nu_j nubar_j
    nu_e = [((L, R, 0), 'e'),                # nu e- -> nu e-
            ((R, L, 0), 'e'),                # nu e+ -> nu e+
            ((0, L, R), 'nu')]               # nu nubar -> e- e+  (target: the antineutrino)
    return nu_e, nu_nu

def rate(E1, coeffs, blocking, n=400000):
    """Gamma/(G_F^2 T^4 E1) for one process: MC over the target momentum and the CM scattering direction."""
    A, B, C = coeffs
    # target energy E2 ~ E2^2 f(E2): sample from Gamma(3) and reweight to FD (or keep Boltzmann)
    E2 = rng.gamma(3.0, 1.0, n)
    w2 = (fd(E2)/np.exp(-E2)) if blocking else np.ones(n)
    norm = 2.0 if not blocking else 2.0          # int E^2 e^{-E} dE = 2 (the weights carry the FD/Boltzmann ratio)
    cth = rng.uniform(-1, 1, n)                  # angle between p1 and p2 (plasma frame)
    p1 = np.stack([np.zeros(n), np.zeros(n), np.full(n, E1)], 1)
    sth = np.sqrt(1 - cth**2); ph = rng.uniform(0, 2*np.pi, n)
    p2 = E2[:, None]*np.stack([sth*np.cos(ph), sth*np.sin(ph), cth], 1)
    P = p1 + p2; Etot = E1 + E2
    s = Etot**2 - np.sum(P*P, 1)
    ok = s > 1e-12
    beta = P/Etot[:, None]; b2 = np.sum(beta**2, 1); gam = 1/np.sqrt(1 - np.clip(b2, 0, 1 - 1e-15))
    # p1 in the CM frame
    def boost(E, p, beta, gam, sign):
        bp = np.sum(beta*p, 1)
        b2 = np.sum(beta**2, 1)
        coef = np.where(b2 > 0, (gam - 1)*bp/np.where(b2 > 0, b2, 1), 0.0)
        Ep = gam*(E - sign*bp)
        pp = p + (coef - sign*gam*E)[:, None]*beta
        return Ep, pp
    E1c, p1c = boost(np.full(n, E1), p1, beta, gam, +1)
    k = np.sqrt(np.maximum(s, 0))/2
    # final direction uniform in the CM frame
    c3 = rng.uniform(-1, 1, n); s3 = np.sqrt(1 - c3**2); f3 = rng.uniform(0, 2*np.pi, n)
    n3 = np.stack([s3*np.cos(f3), s3*np.sin(f3), c3], 1)
    p3c = k[:, None]*n3; p4c = -p3c
    E3, p3 = boost(k, p3c, beta, gam, -1)
    E4, p4 = boost(k, p4c, beta, gam, -1)
    dot = lambda Ea, pa, Eb, pb: Ea*Eb - np.sum(pa*pb, 1)
    E1v = np.full(n, E1)
    p12 = dot(E1v, p1, E2, p2); p13 = dot(E1v, p1, E3, p3); p14 = dot(E1v, p1, E4, p4)
    M2 = 32*(A*p12**2 + B*p14**2 + C*p13**2)
    block = (1 - fd(E3))*(1 - fd(E4)) if blocking else 1.0
    integrand = np.where(ok, w2*M2*block/(4*E1*E2), 0.0)
    # Gamma = (1/(2 pi^2)) int E2^2 f2 dE2 (1/2) int dcos ... ; int dOmega*/(32 pi^2) = (4 pi/(32 pi^2)) <.>
    val = (norm/(2*np.pi**2))*(4*np.pi/(32*np.pi**2))*np.mean(integrand)
    return val/E1

if __name__ == '__main__':
    xs = [0.5, 1.0, 2.0, 3.15, 5.0, 8.0, 12.0, 20.0]
    out = {}
    for fl in ('e', 'mu'):
        nu_e, nu_nu = processes(fl)
        for variant in ('MB', 'FD'):
            blocking = variant == 'FD'
            ce = [sum(rate(x, c, blocking) for c, tgt in nu_e) for x in xs]
            cn = [sum(rate(x, c, blocking) for c, tgt in nu_nu) for x in xs]
            key = f"{fl}_{variant}"
            out[key] = dict(x=xs, nu_e=ce, nu_nu=cn)
            print(key, 'nu-e:', ' '.join(f'{v:.4f}' for v in ce), '| nu-nu:', ' '.join(f'{v:.4f}' for v in cn), flush=True)
    # analytic check (Boltzmann statistics, no blocking): Gamma/(G_F^2 T^4 E) = (4/(3 pi)) (A + B/3 + C/3) rho2, rho2 = 3/pi^2
    rho2 = 3/np.pi**2
    for fl in ('e', 'mu'):
        nu_e, nu_nu = processes(fl)
        an_e = sum((4/(3*np.pi))*(c[0] + c[1]/3 + c[2]/3)*rho2 for c, t in nu_e)
        an_n = sum((4/(3*np.pi))*(c[0] + c[1]/3 + c[2]/3)*rho2 for c, t in nu_nu)
        print(f'analytic MB {fl}: nu-e {an_e:.4f}  nu-nu {an_n:.4f}')
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weak_rates.json'), 'w'))
