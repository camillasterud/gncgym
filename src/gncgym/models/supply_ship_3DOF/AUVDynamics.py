import numpy as np
from gncgym.simulator.angle import Angle
from .gncUtilities import Rzyx

def make_auv_dynamics_block():
    from .modelConstantsAUV import M_inv, D, B

    def model_dynamics(state, f):
        """
        The dynamic model of the supply ship.
        :param state: [eta, nu] = [x,y,psi,u,v,r]
        :param f: [u_des, delta_r]
        :return: state_dot
        """

        # Unpack state values and construct the velocity vector nu
        _, _, psi, u, v, r = state
        nu = np.array([u, v, r])

        eta_dot = Rzyx(0, 0, Angle(psi)).dot(nu)
        nu_dot = M_inv.dot(B(u).dot(f) - D(u, v, r).dot(nu))
        return np.concatenate([eta_dot, nu_dot])

    # Returns a reference to the dynamics function, that can be called to
    # obtain the time-derivative of the state.
    return model_dynamics


