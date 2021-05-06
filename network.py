from __future__ import annotations

from queue import PriorityQueue

import numpy as np
import enum
import math

from utils import WrsnParameters
from utils import NetworkInput, Point, logger
from utils import dist, transmission_energy, energy_consumption

class NodeType(enum.Enum):
    BS = 0
    SN = 1
    RN = 2
    TG = 3


class Node():
    """Node.
    """

    def __init__(self, position, _id, _type=NodeType.SN, is_active=True, forwarding=False,
                 r_c=80, r_s=40):
        self.position = position
        self.id = _id
        self.adj = list()
        self.is_active = is_active
        self.type = _type
        self.forwarding = forwarding
        self.r_c = r_c
        self.r_s = r_s

    def __repr__(self):
        return str((self.id, self.position))

    def __str__(self):
        return str((self.id, self.position))

    def is_connected_to(self, other: Node):
        type_set = set((self.type, other.type)) 

        if NodeType.TG in type_set and NodeType.SN not in type_set:
            return False

        communication_range = self.r_s if NodeType.TG in type_set else self.r_c

        if dist(self.position, other.position) <= communication_range:
            return True
        else:
            return False


class Sensor(Node):
    """Sensor.
    """

    def __init__(self, position, battery_cap, _id, no_targets=0, **kwargs):
        self.battery_cap = battery_cap
        self.cur_energy = battery_cap
        self.ecr = None
        self.no_targets = no_targets
        super(Sensor, self).__init__(position, _id,
                                     NodeType.SN, **kwargs)

    def get_state(self):
        return np.array([self.position.x,
                         self.position.y,
                         self.battery_cap,
                         self.no_targets,
                         self.cur_energy,
                         self.ecr],
                        dtype=np.float32)

    def reset(self):
        """reset.
        """
        self.cur_energy = self.battery_cap
        self.ecr = None
        self.activate()

    def deactivate(self):
        """deactivate.
        """
        self.is_active = False

    def activate(self):
        """activate.
        """
        self.is_active = True


class WRSNNetwork():
    """WRSNNetwork.
    """

    def __init__(self, inp: NetworkInput, wp: Config=WrsnParameters):
        self.inp = inp
        self.wp = wp
        self.num_sensors = inp.num_sensors
        self.num_targets = inp.num_targets
        self.num_charging_points = inp.num_charging_points
        self.num_nodes = self.num_sensors + self.num_targets + 1

        self.sink = Node(inp.sink, 0, NodeType.BS)
        self.sensors = [Sensor(sn, wp.E_s, i, r_c=wp.r_c, r_s=wp.r_s)
                        for i, sn in enumerate(inp.sensors, 1)]
        self.targets = [Node(tg, i, NodeType.TG)
                        for i, tg in enumerate(inp.targets, self.num_sensors+1)]
        self.nodes = list([self.sink, *self.sensors, *self.targets])

        self.status = {i: False for i in range(
            self.num_sensors+1, self.num_nodes)}

        self.network_lifetime = 0

        self.is_coverage = True
        self.is_connectivity = True
        self.estimated_ecr = None
        self.estimated_lifetime = None
        self.edges = set()

        self.__build_adjacency()
        self.run_estimation()

    def __build_adjacency(self):
        """__build_adjacency.
        """
        for node in self.nodes:
            for other in self.nodes:
                if node is not other and node.is_connected_to(other):   
                    node.adj.append(other)
                    if (other.id, node.id) not in self.edges:
                        self.edges.add((node.id, other.id))
                        if node.type == NodeType.TG:
                            other.no_targets += 1
                        elif other.type == NodeType.TG:
                            node.no_targets += 1

    def __check_health(self):
        changed = False
        for sn in self.sensors:
            if not sn.is_active and sn.cur_energy >= self.wp.p_auto_start_threshold * sn.battery_cap:
                sn.activate()
                changed = True
            if sn.is_active and (sn.cur_energy <= self.wp.p_sleep_threshold * sn.battery_cap or
                                 (sn.ecr and sn.cur_energy / sn.ecr < 1.0)):
                sn.deactivate()
                changed = True
        return changed

    def __estimate_topology(self):
        """__dijkstra.
        """
        trace = np.full(self.num_nodes, -1, dtype=int)
        d = np.full(self.num_nodes, np.inf, dtype=np.float32)
        d[0] = 0.0
        q = PriorityQueue()
        q.put((d[0], 0))

        while not q.empty():
            du, u = q.get()

            # BS and TG do not have forwarding function
            if d[u] != du or self.nodes[u].type is NodeType.TG \
                    or (self.nodes[u].type is NodeType.BS and du > 0):
                continue

            for neighbor in self.nodes[u].adj:
                if not neighbor.is_active:
                    continue
                v = neighbor.id
                duv = du + dist(self.nodes[u].position, self.nodes[v].position)
                if d[v] > duv:
                    d[v] = duv
                    trace[v] = u
                    q.put((d[v], v))

        # check if the network covers all targets or not
        if any(trace[self.num_sensors+1:] == -1):
            self.is_coverage = False

        return trace

    def __estimate_ecr(self, trace):
        """__estimate energy consumption rate.
        """
        # number of packets each sensor forwarding
        eta = np.zeros(self.num_sensors + 1)

        for i in range(1, self.num_sensors+1):
            u = i
            while u > 0 and trace[u] != -1:
                u = trace[u]
                eta[u] += 1

        ecr = np.zeros(self.num_sensors + 1)
        for u in range(1, self.num_sensors + 1):
            if trace[u] == -1:
                ecr[u] = 0
            else:
                pu = trace[u]
                d = dist(self.nodes[u].position, self.nodes[pu].position)
                ecr[u] = energy_consumption(eta[u], 1, d, wp=self.wp)
            self.nodes[u].ecr = ecr[u]

        return ecr

    @property
    def active_status(self):
        return np.array([self.nodes[i].is_active for i in range(self.num_sensors + 1)])

    @property
    def aggregated_ecr(self):
        return np.sum(self.estimated_ecr)

    @property
    def node_failures(self):
        return self.num_sensors - np.sum(self.active_status) + 1

    def reset(self):
        """reset.
        """
        for sn in self.sensors:
            sn.reset()
        self.network_lifetime = 0

        self.is_coverage = True
        self.is_connectivity = True
        self.estimated_ecr = None
        self.estimated_lifetime = None
        self.run_estimation()

    def run_estimation(self):
        """run_estimation.
            Simulate network communication, this function omits real communication,
            Only simulate sequential results in base station
        """
        self.__check_health()
    
        while True:
            self.routing_path = self.__estimate_topology()
            self.estimated_ecr = self.__estimate_ecr(self.routing_path)
            self.estimated_lifetime = np.zeros(self.num_sensors + 1)
            for sn in self.sensors:
                if sn.is_active and self.estimated_ecr[sn.id] > 1e-8:
                    self.estimated_lifetime[sn.id] = sn.cur_energy / \
                        self.estimated_ecr[sn.id]
            if not self.__check_health():
                break
    
    def estimate_trans_time(self):
        self.run_estimation()
        cur_energy = np.array([sn.cur_energy for sn in self.sensors])
        battery_cap = np.array([sn.battery_cap for sn in self.sensors])
        ecr = np.array([sn.ecr for sn in self.sensors])
        requested = (cur_energy <= battery_cap * self.wp.p_request_threshold)
        threshold = np.zeros_like(cur_energy)
        threshold = self.wp.p_sleep_threshold * battery_cap * requested + \
            self.wp.p_request_threshold * battery_cap * np.logical_not(requested) 
        trans_time = (cur_energy - threshold) / ecr
        return np.min(trans_time[trans_time > 1.0])

    def t_step(self, t: int, charging_sensors=None):
        """ simulate network running for t seconds.

        Parameters
        ----------
        t : int
            simulated time
        charging_sensors : dict()
            dictionary of charing sensors where key is sensor's id and value is energy charning rate 
        """
        if t <= 0 or not self.is_coverage:
            return 0

        if charging_sensors is None:
            charging_sensors = dict()

        # we assume that packets are generated once each second,
        # and transmission time is trivial
        frac, whole = math.modf(self.network_lifetime)
        # self.network_lifetime = whole
        t += frac
        simulated_time = 0

        while t >= 1.0:
            # self.run_estimation()
            # do not consider lifetime of base station
            active = np.logical_and(self.active_status, 
                                    self.routing_path[:self.num_sensors + 1] != -1)

            min_lifetime = np.min(
                self.estimated_lifetime[active], initial=np.inf)

            time_jump = int(min(min_lifetime, t))
            if time_jump == 0:
                logger.error(active)
                logger.error(t, min_lifetime)
                logger.error(self.estimated_lifetime)
                self.inp.to_file('data/bug/net1.inp')
                raise ValueError('Unexpected zero-value in estimated_lifetime')

            # run network for time_jump seconds
            for sn in self.sensors:
                if sn.id in charging_sensors:
                    sn.cur_energy -= (self.estimated_ecr[sn.id] -
                                      charging_sensors[sn.id]) * time_jump
                elif sn.is_active:
                    sn.cur_energy -= self.estimated_ecr[sn.id] * time_jump

                sn.cur_energy = max(min(sn.cur_energy, sn.battery_cap), 0.0)

            # we should run estimation after network changing
            self.run_estimation()

            # network is not coverage anymore,
            # further running is not counted in network_lifetime
            if self.is_coverage:
                simulated_time += time_jump
            else:
                break
            t -= time_jump

        if self.is_coverage:
            simulated_time += t - frac
            for snid, mu in charging_sensors.items():
                sn = self.nodes[snid]
                sn.cur_energy += mu * (t - frac)
                sn.cur_energy = max(min(sn.cur_energy, sn.battery_cap), 0.0)

        self.network_lifetime += simulated_time
        return simulated_time

    def get_state(self):
        state = []
        for sn in self.sensors:
            state.append(sn.get_state())
        return np.array(state)



if __name__ == '__main__':
    inp = NetworkInput.from_file('net1.inp')
    net = WRSNNetwork(inp)
    net.t_step(5000, charging_sensors={2: 1})
