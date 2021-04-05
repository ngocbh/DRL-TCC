import os
import yaml
from yaml import Loader

class Config():
    __dictpath__ = ''

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        for key, value in d.items():
            setattr(cls, key, value)

    @classmethod
    def from_file(cls, filepath, dictpath=None):
        with open(filepath, mode='r') as f:
            all_config = yaml.load(f, Loader=Loader)

        dictpath = dictpath or cls.__dictpath__
        my_config = all_config
        try:
            for key in dictpath.split('.'):
                if key != '':
                    my_config = my_config[key]
        except:
            my_config = None

        cls.from_dict(my_config)

    @classmethod
    def to_dict(cls):
        d = dict(cls.__dict__)
        for key, value in cls.__dict__.items():
            if '__' in key and key not in ['__dictpath__', '__doc__']:
                d.pop(key, None)
        return d

    @classmethod
    def to_file(cls, filepath, dictpath=None):
        d = cls.to_dict()
        dictpath = dictpath or d['__dictpath__']
        my_config = d
        for key in reversed(dictpath.split('.')):
            if key != '':
                my_config = {key: my_config}

        config = {}
        if os.path.exists(filepath):
            with open(filepath, mode='r') as file:
                config = yaml.load(file, Loader=Loader)

        config.update(my_config)

        with open(filepath, mode='w') as file:
            yaml.dump(my_config, file)

class WrsnParameters(Config):
    __dictpath__ = 'wp'
    # width
    W = 500
    H = 500
    sink = (W/2, H/2, 0)
    depot = (0, 0, 0)
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
    mu = 20

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
    k_bit = 8000000

    # hop constraint
    hop = 12

class DrlParameters(Config):
    __dictpath__ = 'dp'
    # input sizes
    MC_STATIC_SIZE = 4
    MC_DYNAMIC_SIZE = 3
    MC_INPUT_SIZE = MC_STATIC_SIZE + MC_DYNAMIC_SIZE
    SN_STATIC_SIZE = 3
    SN_DYNAMIC_SIZE = 2
    SN_INPUT_SIZE = SN_STATIC_SIZE + SN_DYNAMIC_SIZE

    # Neural network parameters
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1
    DROPOUT = 0.2

    gamma = 0.9

if __name__ == '__main__':
    WrsnParameters.from_file('./configs/config.yml')
    print(WrsnParameters.to_file('./configs/config2.yml'))
