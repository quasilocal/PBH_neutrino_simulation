import numpy as np, sys
sys.path.insert(0,'.')
from linear_nu import evolve
from scipy.special import erfc, erfcinv
r0=evolve(0.0)
print("ratio Delta_nu/Delta_fluid at fixed zeta, vs k*tau (horizon-crossing / transfer-function proxies):")
for R in [0.409,0.488]:
    r=evolve(R); t=r['t']
    row=[f"{tt}: {np.interp(tt,t,r['Delta'])/np.interp(tt,t,r0['Delta']):.4f}" for tt in [0.5,1,2,2.7,3.5,4.5]]
    late=np.max(np.abs(r['Delta'][t>25]))/np.max(np.abs(r0['Delta'][t>25]))
    print(f"  R={R}: "+", ".join(row)+f"; late acoustic amplitude ratio {late:.3f}")
# abundance impact: beta ~ erfc(nu/sqrt2), nu = delta_c/sigma, threshold scaled by factor f at fixed sigma
print("\nabundance ratio beta'/beta when the threshold is multiplied by f at fixed P_zeta:")
for beta in [1e-6,1e-9,1e-12,1e-15]:
    nu=np.sqrt(2)*erfcinv(beta)
    out=[f"f={f}: {erfc(f*nu/np.sqrt(2))/beta:.2e}" for f in [0.942,1.036,1.10,1.115]]
    print(f"  beta={beta:.0e} (nu={nu:.2f}): "+"; ".join(out))
