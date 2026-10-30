import numpy as np, time
from ms_fluid import rhs, derived, initial_data, make_grid, rho_b

def run(delta_m, N=800, rmax=9.0, beta=3.0, eps0=0.1, cfl=0.4, cQ=2.0, eps_max=30.0,
        verbose=False, snap_eps=(), qshape=1.0, cH=0.005):
    r = make_grid(N, rmax, beta)
    t0, R, U, rho = initial_data(r, delta_m, eps0, 0.0, qshape)
    e = rho.copy(); z = np.zeros(N)
    t = t0; cs = 1/np.sqrt(3.0)
    dR = np.zeros(N+1); dU = np.zeros(N+1); de = np.zeros(N)
    peakC = -1.0; hist = []; snaps = {}
    step = 0; status = 'undecided'
    snap_eps = list(snap_eps)
    while True:
        M, Gam, p, Ac, Ae = derived(t, R, U, e, z, z, z, 0.0, cQ)
        epsn = 2*np.sqrt(t*t0)
        # diagnostics
        twoMR = 2*M[1:]/R[1:]
        C = 2*(M[1:] - (4*np.pi/3)*rho_b(t)*R[1:]**3)/R[1:]
        iC = np.argmax(C); peakC = max(peakC, C[iC])
        if step % 20 == 0:
            hist.append((t, epsn, C[iC], r[1+iC], e[0]/rho_b(t)))
        if snap_eps and epsn >= snap_eps[0]:
            snaps[snap_eps.pop(0)] = dict(t=t, R=R.copy(), U=U.copy(), e=e.copy(), M=M.copy())
        bh = np.where((twoMR >= 1.0) & (U[1:] < 0))[0]
        if bh.size:
            status = 'BH'; break
        if epsn > 3.0 and C[iC] < 0.5*peakC:
            status = 'dispersed'; break
        if epsn > eps_max: break
        # time step
        dRc = np.diff(R); Gc = 0.5*(Gam[:-1]+Gam[1:])
        dt = cfl*np.min(dRc/(np.abs(Gc)*Ac*cs))
        dt = min(dt, cH*2*t)          # H dt <= cH: background truncation error must stay << eps0^2
        # SSP RK3
        M1 = rhs(t, R, U, e, z, z, z, 0.0, cQ, dR, dU, de)
        R1 = R + dt*dR; U1 = U + dt*dU; e1 = e + dt*de
        rhs(t+dt, R1, U1, e1, z, z, z, 0.0, cQ, dR, dU, de)
        R2 = 0.75*R + 0.25*(R1 + dt*dR); U2 = 0.75*U + 0.25*(U1 + dt*dU); e2 = 0.75*e + 0.25*(e1 + dt*de)
        rhs(t+0.5*dt, R2, U2, e2, z, z, z, 0.0, cQ, dR, dU, de)
        R = R/3 + 2/3*(R2 + dt*dR); U = U/3 + 2/3*(U2 + dt*dU); e = e/3 + 2/3*(e2 + dt*de)
        t += dt; step += 1
        if np.any(e <= 0) or not np.all(np.isfinite(U)):
            status = 'crash'; break
    return dict(status=status, t=t, eps=2*np.sqrt(t*t0), steps=step, peakC=peakC,
                hist=np.array(hist), snaps=snaps, r=r)

if __name__ == '__main__':
    for dm in [0.40, 0.45, 0.55, 0.60]:
        t1 = time.time(); out = run(dm)
        h = out['hist']
        print(f"delta_m={dm}: {out['status']} at eps={out['eps']:.3f}, steps={out['steps']}, peakC={out['peakC']:.3f}, "
              f"time {time.time()-t1:.1f}s; C_max(eps) samples:",
              ' '.join(f"{x[1]:.2f}:{x[2]:.3f}" for x in h[::max(1,len(h)//8)]))
