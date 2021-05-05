import math

from utils.parameters import WrsnParameters

def transmission_energy(k, d, wp=WrsnParameters):
    """transmission_energy.

    Parameters
    ----------
    k :
        the number of bits in a packet
    d :
        the distance to the parent node
    """
    d0 = math.sqrt(wp.e_fs / wp.e_mp)
    if d <= d0:
        return k * wp.e_elec + k * wp.e_fs * (d ** 2)
    else:
        return k * wp.e_elec + k * wp.e_mp * (d ** 4)

def energy_consumption(x, y, d, wp=WrsnParameters):
    """energy_consumption.

    Parameters
    ----------
    x : int
        the number of received packets (forwarding purpose)
    y : binary value (0, 1)
        1 if it has sensing data (sensor), 0 otherwise 
    d : float
        distance to the parent node
    """
    e_t = transmission_energy(wp.k_bit, d)
    # e_r = x * wc.k_bit * (wc.e_elec + wc.e_da) + y * wc.k_bit * wc.e_da
    e_r = wp.k_bit * wp.e_elec
    e = x * e_r + (x + y) * e_t
    return e