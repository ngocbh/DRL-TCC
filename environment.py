import gym
import math
import numpy as np
from gym import spaces, logger
from gym.utils import seeding


from utils import WrsnParameters as wp
from utils import NetworkInput, Point
from utils import energy_consumption, dist
from wrsn_network import WRSNNetwork


class MobileCharger():
    """MobileCharger.
    """

    def __init__(self, position, battery_cap, velocity, ecr_move, e_charge):
        self.depot = position
        self.cur_position = position
        self.battery_cap = battery_cap
        self.velocity = velocity
        self.ecr_move = ecr_move
        self.e_charge = e_charge
        self.cur_energy = battery_cap
        self.is_active = True
        self.lifetime = 0

    def get_state(self):
        return np.array([self.cur_position.x,
                         self.cur_position.y,
                         self.cur_energy,
                         self.battery_cap,
                         self.ecr_move,
                         self.e_charge,
                         self.velocity],
                        dtype=np.float32)

    def reset(self):
        self.cur_position = self.depot
        self.cur_energy = self.battery_cap
        self.activate()
        self.lifetime = 0

    def deactivate(self):
        """deactivate.
        """
        self.is_active = False

    def activate(self):
        """activate.
        """
        self.is_active = True

    def move(self, dest: Point):
        d1 = dist(self.cur_position, dest)
        d2 = min(d1, self.cur_energy / self.ecr_move)
        t1 = d1 / self.velocity
        t2 = d2 / self.velocity
        e = d2 * wp.ecr_move
        self.cur_position = Point(self.cur_position.x + t2/t1 * (dest.x - self.cur_position.x),
                                  self.cur_position.y + t2/t1 * (dest.y - self.cur_position.y),
                                  self.cur_position.z + t2/t1 * (dest.z - self.cur_position.z)) 

        self.cur_energy -= e
        if self.cur_energy <= 0:
            self.deactivate()

        self.lifetime += t2

        return (t2, d2)

    def charge(self):
        pass

    def recharge(self):
        pass


class WRSNEnv(gym.Env):
    """WRSNEnv.
    Description:
        A simulation of Wireless Rechargable Sensor Network

    Observation:
        Type: Tuple(Box(7), Box(num_sensors * 5))
        Box(7): Observation of MC, the first 3 values are dynamic,
                and 4 next values are static
            (x_coor, y_coor, current_energy, 
            battery_capacity, moving_energy_consumption_rate, 
            charging_energy_rate, velocity)
        Box(num_sensors * 5): Box[i][:] for observation of sensor ith
                              the first 3 values are static and
                              the rest of them is dynamic
            (x_coor, y_coor, battery_capacity, 
            current_energy, energy_consumption_rate)

    Actions:
        Type: Discrete(num_sensors + 1)
        0 : Run MC back to depot and recharging
        i : Run MC to sensor i and charging sensor ith 

    Reward:
        (t, d)
        t: lifetime
        d: moving distance of mobile charger

    Starting state:
        All sensors and MC are initialized with full battery
        Other values are inferred from Network

    Episode Termination:
        MC is exhausted and cannot come back to depot to recharge
        Network is not coverage ( not cover all targets )

    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self, inp: NetworkInput, seed=None):
        self.inp = inp
        self.depot = inp.depot
        self.charging_points = inp.charging_points
        self.action_dest = [inp.depot, *inp.charging_points]
        self.mc = MobileCharger(
            inp.depot, wp.E_mc, wp.v_mc, wp.ecr_move, wp.e_charge)
        self.net = WRSNNetwork(inp)

        max_ecr = energy_consumption(inp.num_sensors, 1, wp.r_c)
        high_s_row = np.array([inp.W,
                               inp.H,
                               wp.E_s,
                               wp.E_s,
                               max_ecr],
                              dtype=np.float32)
        high_s = np.tile(high_s_row, (inp.num_sensors, 1))
        low_s = np.zeros((inp.num_sensors, 5), dtype=np.float32)

        high_mc = np.array([inp.W,
                            inp.H,
                            wp.E_mc,
                            wp.E_mc,
                            wp.ecr_move,
                            wp.e_charge,
                            wp.v_mc],
                           dtype=np.float32)
        low_mc = np.zeros(7, dtype=np.float32)

        self.action_space = spaces.Discrete(self.net.num_sensors + 1)
        self.observation_space = spaces.Tuple((spaces.Box(low_mc, high_mc, dtype=np.float32),
                                               spaces.Box(low_s, high_s, shape=(inp.num_sensors, 5),
                                                          dtype=np.float32)))

        self.seed(seed)
        self.state = None
        self.viewer = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # 2 phases: move to dest and charge (or recharge)
        # phase 1: move MC to dest
        t_mc, d_mc = self.mc.move(self.action_dest[action])
        print(t_mc, d_mc)


    def reset(self):
        self.net.reset()
        self.mc.reset()
        self.state = (self.mc.get_state(), self.net.get_state())
        self.steps_beyond_done = None
        return self.state

    def render(self, mode):
        raise NotImplementedError


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    inp = NetworkInput.from_file('data/test/NIn1.json')
    env = WRSNEnv(inp)
    env.reset()
    env.step(1)
