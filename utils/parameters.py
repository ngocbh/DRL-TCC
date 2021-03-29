
class WrsnParameters:
    num_mc = 1 # number of mobile charger
    r_c = 80 # communication range (m)
    r_s = 40 # sensing range (m)
    E_s = 6000 # capacity of sensor (J)
    v_mc = 5 # velocity of mc (m/s)
    ecr_move = 50 # energy consumption per unit distance of MC (J/m)
    e_charge = 100 # energy recharging rate of MC at depot (J/s)
    E_mc = 1 * 1e6# capacity of mc (J)
    mu = 20 # Charging rate (W)

    lamb = 36.0 # transmission parameter
    beta = 30.0 # transmission parameter

    # Unit: J
    e_elec = 50 * 1e-9
    e_fs = 10 * 1e-12
    e_mp = 0.0013 * 1e-12
    e_da = 5 * 1e-12

    # Num of bits
    k_bit = 4000000

    # hop constraint
    hop = 12

class DrlParameters:
    gamma = 0.9
