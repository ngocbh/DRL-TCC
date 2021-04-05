import gym
import math
import numpy as np
import pyglet
from gym import spaces, logger
from gym.utils import seeding


from utils import WrsnParameters as wp
from utils import NetworkInput, Point
from utils import energy_consumption, dist, normalize
from network import WRSNNetwork


class MobileCharger():
    """MobileCharger.
    """

    def __init__(self, position, battery_cap, velocity, ecr_move, ecr_charge):
        self.depot = position
        self.cur_position = position
        self.battery_cap = battery_cap
        self.velocity = velocity
        self.ecr_move = ecr_move
        self.ecr_charge = ecr_charge
        self.cur_energy = battery_cap
        self.is_active = True
        self.lifetime = 0

    def get_state(self):
        return np.array([self.cur_position.x,
                         self.cur_position.y,
                         self.cur_energy,
                         self.battery_cap,
                         self.ecr_move,
                         self.ecr_charge,
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
        src = self.cur_position
        d1 = dist(src, dest)
        d2 = min(d1, self.cur_energy / self.ecr_move)
        t1 = d1 / self.velocity
        t2 = d2 / self.velocity
        if t1 == 0:
            return (0, 0)
        e = d2 * wp.ecr_move
        self.cur_position = Point(src.x + t2/t1 * (dest.x - src.x),
                                  src.y + t2/t1 * (dest.y - src.y),
                                  src.z + t2/t1 * (dest.z - src.z))

        self.cur_energy -= e
        if self.cur_energy <= 0:
            self.deactivate()

        self.lifetime += t2

        return (t2, d2)

    def charge(self, ce, te, ecr, mu):
        # If charging rate = energy consumption rate, then charge it until exhausted
        if ecr == mu:
            return self.cur_energy / mu

        # if charing rate < energy consumption rate, then target energy is zero
        if mu - ecr <= 0:
            te = 0

        # n is floor of maximum charging time in order to reach target energy
        # this function's considering discrete energy consumption of sensors
        # it means sensors will be dissipated ecr J once each second
        # other components of energy model such as idle, sleep, sensing energy is omitted
        # note that this formulation is still not correct since it 
        # does not consider decimal fraction of current network lifetime
        # however, to keep it simple, we omitted it
        # as a consequence, sometimes, mc leaves the sensor not being full charged
        n = int((te - ce) / (mu - ecr))
        alpha = te - ce - n *  (mu - ecr)
        t = n + alpha

        if self.cur_energy > mu * t:
            self.cur_energy -= mu * t 
            return t
        else:
            self.cur_energy = 0.0
            self.deactivate()
            return self.cur_energy / mu

    def recharge(self):
        t = (self.battery_cap - self.cur_energy) / self.ecr_charge
        self.cur_energy = self.battery_cap
        return t


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
        Box(num_sensors, 5): Box[i, :] for observation of sensor ith
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

    def __init__(self, inp: NetworkInput, seed=None, normalize=False):
        self.inp = inp
        self.world_width = inp.W
        self.world_height = inp.H
        self.depot = inp.depot
        self.charging_points = inp.charging_points
        self.action_dest = [inp.depot, *inp.charging_points]
        self.mc = MobileCharger(
            inp.depot, wp.E_mc, wp.v_mc, wp.ecr_move, wp.ecr_charge)
        self.net = WRSNNetwork(inp)
        self.normalize = normalize

        max_ecr = energy_consumption(inp.num_sensors, 1, wp.r_c)
        high_s_row = np.array([inp.W,
                               inp.H,
                               wp.E_mc,
                               wp.E_mc,
                               max_ecr],
                              dtype=np.float32)
        self.high_s = np.tile(high_s_row, (inp.num_sensors, 1))
        self.low_s = np.zeros((inp.num_sensors, 5), dtype=np.float32)

        self.high_mc = np.array([inp.W,
                            inp.H,
                            wp.E_mc,
                            wp.E_mc,
                            wp.ecr_move,
                            wp.ecr_charge,
                            wp.v_mc],
                           dtype=np.float32)
        self.low_mc = np.zeros(7, dtype=np.float32)

        self.action_space = spaces.Discrete(self.net.num_sensors + 1)
        self.observation_space = spaces.Tuple((spaces.Box(self.low_mc, self.high_mc, dtype=np.float32),
                                               spaces.Box(self.low_s, self.high_s, shape=(inp.num_sensors, 5),
                                                          dtype=np.float32)))

        self.seed(seed)
        self.state = None
        self.viewer = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(self.np_random.randint(1000))
        self.observation_space.seed(self.np_random.randint(1000))
        return [seed]

    def step(self, action):
        """step.
        Accepts an action and returns a tuple (observation, reward, done, info).

        Parameters
        ----------
            action (object): move to a sensor (or depot if action = 0) and charge it (or recharge)

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward_t, reward_d = 0.0, 0.0

        # 2 phases: move to dest and charge (or recharge)
        # phase 1: move MC to dest
        t1_mc, d_mc = self.mc.move(self.action_dest[action])
        # simultaneously simulate the network running in t_1 seconds
        t1_net = self.net.t_step(t1_mc, charging_sensors=None)
        
        reward_t += min(t1_mc, t1_net)
        reward_d += d_mc

        if not self.net.is_coverage:
            pass
        # phase 2: charge or recharge
        elif action == 0:
            # recharging
            t2_mc = self.mc.recharge()
            t2_net = self.net.t_step(t2_mc, charging_sensors=None)

            reward_t += min(t2_mc, t2_net)
        else:
            # charge sensor ith
            sn = self.net.nodes[action]

            # if sensor is exhausted, precharge p percent first and reregister sensor to network
            if not sn.is_active:
                t2_mc = self.mc.charge(sn.cur_energy, 
                                     sn.battery_cap * wp.p_start_threshold,
                                     sn.ecr,
                                     wp.mu)
                t2_net = self.net.t_step(t2_mc, charging_sensors={action: wp.mu})
                reward_t += min(t2_mc, t2_net)

            # continue charging until getting full battery
            if self.net.is_coverage:
                t3_mc = self.mc.charge(
                    sn.cur_energy, sn.battery_cap, sn.ecr, wp.mu)

                t3_net = self.net.t_step(t3_mc, charging_sensors={action: wp.mu})
                reward_t += min(t3_mc, t3_net)

        # if mc is exhausted, cannot improve the network lifetime anymore,
        # fast forward network simulation and stop game
        if not self.mc.is_active and self.net.is_coverage:
            reward_t += self.net.t_step(np.inf, charging_sensors=None)

        self.state = (self.mc.get_state(), self.net.get_state())

        done = bool(
            not self.net.is_coverage
            or not self.mc.is_active
        )

        if not done:
            reward = (reward_t, reward_d)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = (reward_t, reward_d)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = (0, np.inf)

        return (self.state, reward, done, {})

    def reset(self):
        self.net.reset()
        self.mc.reset()
        self.state = (self.mc.get_state(), self.net.get_state())
        self.steps_beyond_done = None
        return self.state

    def get_normalized_state(self):
        mc_state, sensors_state = self.state
        return (normalize(mc_state, self.low_mc, self.high_mc),
                normalize(sensors_state, self.low_s, self.high_s))

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        scale = screen_width / self.world_width

        sn_width, sn_height, sn_color  = 20, 10, (0.9, 0.9, .12)
        mc_width, mc_height, mc_color = 30, 30, (.06, .2, .96) 
        tg_radius, tg_color = 5, (.9, .1, .1)

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.lines = dict()
            self.trans = [None] * self.net.num_nodes
            self.objs = [None] * self.net.num_nodes
            # draw edges
            for u, v in self.net.edges:
                su, sv = self.net.nodes[u], self.net.nodes[v]
                if su.is_active and sv.is_active:
                    sux, suy, _ = su.position
                    svx, svy, _ = sv.position
                    sux, suy = sux * scale, suy * scale
                    svx, svy = svx * scale, svy * scale
                    line = rendering.Line((sux, suy), (svx, svy))
                    self.lines[(u, v)] = line
                    self.viewer.add_geom(line)

            l, r, t, b = -10, 10, 10, -10
            x, y, _ = self.net.sink.position
            x, y = x * scale, y * scale
            sink_obj = rendering.FilledPolygon([(x, y + t), (x + l, y + b), 
                                                (x + r, y + b), (x, y + t)])
            self.viewer.add_geom(sink_obj)
            self.objs[0] = sink_obj

            l, r, t, b = -mc_width / 2, mc_width / 2, mc_height / 2, -mc_height / 2
            mc = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.mctrans = rendering.Transform()
            mc.add_attr(self.mctrans)
            mc.set_color(*mc_color)
            self.viewer.add_geom(mc)

            for sn in self.net.sensors:
                l, r, t, b = -sn_width / 2 , sn_width / 2 , sn_height / 2, -sn_height / 2
                x, y, _ = sn.position
                x, y = x * scale, y * scale

                snb = self.viewer.draw_polyline([(x + l, y + b), (x + l, y + t), 
                                           (x + r, y + t), (x + r, y + b),
                                           (x + l, y + b)])
                self.viewer.add_geom(snb)

                l, r, t, b = -sn_width / 2 + 1, sn_width / 2 - 1, sn_height / 2 - 1, -sn_height / 2 + 1
                sno = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                ep = sn.cur_energy / sn.battery_cap
                sn_scl = (ep, 1)
                r, g, b = min(0.9, 1.8 *  (1 - ep)), min(0.9, 1.8 *  ep), .12
                sn_c = (sn_color[0] * (1 - ep), sn_color[1], sn_color[2])
                x += (1 - sn_scl[0]) * l 
                sntrans = rendering.Transform(translation=(x, y), scale=sn_scl)
                sno.add_attr(sntrans)
                sno.set_color(r, g, b)
                self.viewer.add_geom(sno)

                self.trans[sn.id] = sntrans
                self.objs[sn.id] = sno

            for tg in self.net.targets:
                x, y, _ = tg.position
                x, y = x * scale, y * scale
                circ = self.viewer.draw_circle(tg_radius)
                circ.add_attr(rendering.Transform(translation=(x, y)))
                circ.set_color(*tg_color)
                self.objs[tg.id] = circ
                self.viewer.add_geom(circ)


        if self.state is None:
            return None

        # transform mc
        mc_state, sn_state = self.state
        self.mctrans.set_translation(mc_state[0] * scale, mc_state[1] * scale)

        # transform sns
        for sn in self.net.sensors:
            l, r, t, b = -sn_width / 2 - 1, sn_width / 2 - 1, sn_height / 2 - 1, -sn_height / 2 - 1
            x, y, _ = sn.position
            x, y = x * scale, y * scale
            ep = sn.cur_energy / sn.battery_cap
            sn_scl = (ep, 1)
            r, g, b = min(0.9, 1.8 *  (1 - ep)), min(0.9, 1.8 *  ep), .12
            sn_c = (sn_color[0] * (1 - ep), sn_color[1], sn_color[2])
            x += (1 - sn_scl[0]) * l 
            self.trans[sn.id].set_scale(ep, 1)
            self.trans[sn.id].set_translation(x, y)
            self.objs[sn.id].set_color(r, g, b)


        for u, v in self.net.edges:
            su, sv = self.net.nodes[u], self.net.nodes[v]
            if su.is_active and sv.is_active:
                self.lines[(u, v)].set_color(0, 0, 0)
            else:
                self.lines[(u, v)].set_color(1, 1, 1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    inp = NetworkInput.from_file('data/test/NIn1.json')
    env = WRSNEnv(inp)
    env.reset()
    state, reward, done, _ = env.step(1)
    print(state)
    print(reward, done)
    print(env.mc.cur_position)
    print(env.mc.cur_energy)
