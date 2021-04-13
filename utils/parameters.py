import os
import argparse
import yaml
import torch
from yaml import Loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def to_file(cls, filepath, dictpath=None, mode='merge_cls'):
        """to_file.

        Parameters
        ----------
        filepath :
            filepath
        dictpath :
            dictpath
        mode : str
            mode defines behaviour when file exists
            'new': create new one.
            'merge_cls': merge and prioritize current settings on cls
            'merge_file': merge and prioritize settings on file
        """
        d = cls.to_dict()
        dictpath = dictpath or d['__dictpath__']
        my_config = d
        for key in reversed(dictpath.split('.')):
            if key != '':
                my_config = {key: my_config}

        config = {}
        if os.path.exists(filepath) and mode != 'new':
            with open(filepath, mode='r') as file:
                config = yaml.load(file, Loader=Loader)

        if mode == 'merge_file':
            my_config.update(config)
            config = my_config
        else:
            config.update(my_config)

        with open(filepath, mode='w') as file:
            yaml.dump(config, file)

class WrsnParameters(Config):
    __dictpath__ = 'wp'
    # width
    W = 200
    H = 200
    sink = {'x': W/2,'y': H/2,'z': 0}
    depot = {'x': 0,'y': 0,'z': 0}
    # number of mobile charger
    num_mc = 1 
    # communication range (m)
    r_c = 80 
    # sensing range (m)
    r_s = 40 
    # capacity of sensor (J)
    E_s = 6000
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
    k_bit = 20000000

    # hop constraint
    hop = 12

class DrlParameters(Config):
    __dictpath__ = 'dp'
    # input sizes (do not tune it)
    MC_STATIC_SIZE = 4
    MC_DYNAMIC_SIZE = 3
    MC_INPUT_SIZE = MC_STATIC_SIZE + MC_DYNAMIC_SIZE
    SN_STATIC_SIZE = 4
    SN_DYNAMIC_SIZE = 2
    SN_INPUT_SIZE = SN_STATIC_SIZE + SN_DYNAMIC_SIZE

    # Neural network parameters
    hidden_size = 128
    num_layers = 1
    dropout = 0.2

    # Training parameters
    train_size = 1
    valid_size = 1
    test_size = 1
    batch_size = 1
    num_epoch = 1
    max_step = 1000

    actor_lr = 5e-4
    critic_lr = 5e-4
    max_grad_norm = 2.
    gae_lambda = 0.9
    entropy_coef = 0.01
    gamma = 0.9


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--dump', default='config.yml', type=str)
    parser.add_argument('--mode', default='merge_cls', type=str)

    args = parser.parse_args()
    WrsnParameters.to_file(args.dump, mode=args.mode)
    DrlParameters.to_file(args.dump, mode=args.mode)
