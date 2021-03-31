
class WrsnParameters:
    # number of mobile charger
    num_mc = 1 
    # communication range (m)
    r_c = 80 
    # sensing range (m)
    r_s = 40 
    # capacity of sensor (J)
    E_s = 600
    # When charging a exhausted sensor,
    # allow it joining network when it has at least p percent of battery charged
    p_start_threshold = 0.2
    p_auto_start_threshold = 0.2
    p_sleep_threshold = 0.0
    # velocity of mc (m/s)
    v_mc = 5 
    # energy consumption per unit distance of MC (J/m)
    ecr_move = 50 
    # energy recharging rate of MC at depot (J/s)
    ecr_charge = 2000 
    # capacity of mc (J)
    E_mc = 1 * 1e6
    # Charging rate (W)
    mu = 2

    # transmission parameter
    lamb = 36.0 
    # transmission parameter
    beta = 30.0 

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
