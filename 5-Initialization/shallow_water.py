import numpy as np

def check_field(name, arr, shape):
    """Expand a scalar to the grid shape, or check that an array has that shape."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    return arr


def upwind_x(a, c, dx):
    """Upwind c * da/dx at interior rows of a; zero on the first and last row."""
    d = np.zeros_like(a)
    cc = c[1:-1, :]
    d[1:-1, :] = np.where(cc > 0, cc*(a[1:-1, :] - a[:-2, :])/dx,
                                  cc*(a[2:, :] - a[1:-1, :])/dx)
    return d


def upwind_y(a, c, dy):
    """Upwind c * da/dy at interior columns of a; zero on the first and last column."""
    d = np.zeros_like(a)
    cc = c[:, 1:-1]
    d[:, 1:-1] = np.where(cc > 0, cc*(a[:, 1:-1] - a[:, :-2])/dy,
                                  cc*(a[:, 2:] - a[:, 1:-1])/dy)
    return d


def shallow_water(h0, u0, v0, X, Y, dt, end_hr, f=9.37E-5,
                  g=9.80665, sig_h=1.0, sig_u=1.0, sig_v=1.0, advect=False):
    """
    Shallow water model on an Arakawa C grid.
    x increases south (axis 0), y increases east (axis 1).

      h0, sig_h      : (nx, ny)     at h points  (x[i],        y[j])
      u0, f_u, sig_u : (nx-1, ny)   at u points  (x[i] + dx/2, y[j])
      v0, f_v, sig_v : (nx, ny-1)   at v points  (x[i],        y[j] + dy/2)
      X, Y           : (nx, ny)     h-point coordinates on the image plane (m)

    u is the wind component along x (southward), v along y (eastward).

    advect=False : linear model, dh/dt = -H_bar * div(V). The height pattern
                   cannot translate; only geostrophic adjustment occurs.
    advect=True  : adds upwind momentum advection and flux-form continuity,
                   dh/dt = -div(h V), so the pattern is carried by the wind.

    Held fixed at their initial values: h on all edges, u on the west and
    east columns, v on the north and south rows.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    nx, ny = X.shape
    shape_h, shape_u, shape_v = (nx, ny), (nx-1, ny), (nx, ny-1)

    h0    = check_field('h0',    h0,    shape_h)
    sig_h = check_field('sig_h', sig_h, shape_h)
    u0    = check_field('u0',    u0,    shape_u)
    sig_u = check_field('sig_u', sig_u, shape_u)
    v0    = check_field('v0',    v0,    shape_v)
    sig_v = check_field('sig_v', sig_v, shape_v)

    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]

    n_steps = int(round(end_hr*3600 / dt))

    #base state height (used by the linear continuity equation)
    H_bar = np.mean(h0)

    h = np.zeros((n_steps+1,) + shape_h)
    u = np.zeros((n_steps+1,) + shape_u)
    v = np.zeros((n_steps+1,) + shape_v)

    h[0] = h0
    u[0] = u0
    v[0] = v0

    for n in range(n_steps):
        # start from the current state; edges not updated below stay fixed
        h[n+1] = h[n]
        u[n+1] = u[n]
        v[n+1] = v[n]

        # --- u momentum: f * (v averaged to u points) - g sigma dh/dx ---
        v_at_u = 0.25*(v[n, :-1, :-1] + v[n, :-1, 1:] + v[n, 1:, :-1] + v[n, 1:, 1:])
        dhdx   = sig_u[:, 1:-1] * (h[n, 1:, 1:-1] - h[n, :-1, 1:-1]) / dx
        du     = f*v_at_u - g*dhdx
        if advect:
            v_u = np.zeros(shape_u); v_u[:, 1:-1] = v_at_u
            adv = sig_u * (upwind_x(u[n], u[n], dx) + upwind_y(u[n], v_u, dy))
            du -= adv[:, 1:-1]
        u[n+1, :, 1:-1] = u[n, :, 1:-1] + dt*du

        # --- v momentum, using the new u: -f * (u averaged to v points) - g sigma dh/dy ---
        u_at_v = 0.25*(u[n+1, :-1, :-1] + u[n+1, :-1, 1:] + u[n+1, 1:, :-1] + u[n+1, 1:, 1:])
        dhdy   = sig_v[1:-1, :] * (h[n, 1:-1, 1:] - h[n, 1:-1, :-1]) / dy
        dv     = -f*u_at_v - g*dhdy
        if advect:
            u_v = np.zeros(shape_v); u_v[1:-1, :] = u_at_v
            adv = sig_v * (upwind_x(v[n], u_v, dx) + upwind_y(v[n], v[n], dy))
            dv -= adv[1:-1, :]
        v[n+1, 1:-1, :] = v[n, 1:-1, :] + dt*dv

        # --- continuity at interior h points, using the new winds ---
        if advect:
            # flux form: h upwinded to the cell faces
            hu = np.where(u[n+1, :, 1:-1] > 0, h[n, :-1, 1:-1], h[n, 1:, 1:-1]) \
                 * u[n+1, :, 1:-1] / sig_u[:, 1:-1]
            hv = np.where(v[n+1, 1:-1, :] > 0, h[n, 1:-1, :-1], h[n, 1:-1, 1:]) \
                 * v[n+1, 1:-1, :] / sig_v[1:-1, :]
            div = sig_h[1:-1, 1:-1]**2 * ((hu[1:, :] - hu[:-1, :]) / dx +
                                          (hv[:, 1:] - hv[:, :-1]) / dy)
            h[n+1, 1:-1, 1:-1] = h[n, 1:-1, 1:-1] - dt*div
        else:
            ur = u[n+1] / sig_u
            vr = v[n+1] / sig_v
            div = sig_h[1:-1, 1:-1]**2 * ((ur[1:, 1:-1] - ur[:-1, 1:-1]) / dx +
                                          (vr[1:-1, 1:] - vr[1:-1, :-1]) / dy)
            h[n+1, 1:-1, 1:-1] = h[n, 1:-1, 1:-1] - dt*H_bar*div

    return(h, u, v)

'''
def grid_factors(Xc, Yc, m, rho, phi_0_rad, omega=7.292e-5):
    """Map scale factor sigma and Coriolis parameter f at map points (cm)."""
    lat = np.pi/2 - 2*np.arctan(np.hypot(Xc/m, Yc/m) / (rho*(1 + np.sin(phi_0_rad))))
    return (1 + np.sin(phi_0_rad)) / (1 + np.sin(lat)), 2*omega*np.sin(lat)
'''