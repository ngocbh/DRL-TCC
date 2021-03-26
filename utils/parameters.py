
class WrsnParameters:
    no_mc = 1 # number of mobile charger
    r_c = 25 # communication range (m)
    E_s = 10 # capacity of sensor (J)
    v_mc = 5 # velocity of mc (m/s)
    e_move = 0.01 # energy consumption per unit distance of MC (J/s)
    e_mc = 10 # evergy recharging rate of MC at depot (J/s)
    E_mc = 5000 # capacity of mc (J)
    lamb = 36.0 # transmission parameter
    beta = 30.0 # transmission parameter

class DrlParameters:
    gamma = 0.9
