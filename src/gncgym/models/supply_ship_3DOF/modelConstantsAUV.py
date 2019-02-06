import numpy as np

m = 31
I_zz = 3.45
X_udot = -0.93
Y_vdot = -35.5
Y_rdot = 1.93
N_rdot = -4.88
X_u = 60
Y_v = 60
N_r = 20
X_uu = -1.62
Y_vv = -1310
N_rr = -9.4
Y_rr = 0.632
N_vv = -3.18
Y_uudr = 9.64
N_uudr = -6.15

M_RB = np.diag([m,m,I_zz]);
M_A = -np.array([
    [X_udot, 0, 0],
    [0, Y_vdot, Y_rdot],
    [0, Y_rdot, N_rdot]
    ])
M_inv = np.inv(M_RB+M_A)
C_RB = np.array([
        [0, 0, -m],
        [0, 0, m],
        [m, -m, 0]
        ])
C_A = np.array([
        [0, 0, Y_vdot],
        [0, 0, -X_udot],
        [-Y_vdot, X_udot, 0]
        ])

def C(u, v): 
    return (C_A + C_RB) @ np.array([
                            [0, 0, v],
                            [0, 0, u],
                            [v, u, 0],
                        ])



D_lin = -diag([X_u, Y_v, N_r]);
D_quad = -np.array([
        [X_uu, 0, 0],
        [0, Y_vv, Y_rr],
        [0, N_vv, N_rr],
    ]);

def D(u, v, r):
    return D_quad @ np.diag(np.absolute([u, v, r]))

def B(u):
    return np.array([
            [1, 0],
            [0, Y_uudr*u^2],
            [0, N_uudr*u^2],
        ])

T_max = 40;