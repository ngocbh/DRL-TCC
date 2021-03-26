from queue import PriorityQueue

import numpy as np
import enum

from utils import WrsnParameters as wp
from utils import NetworkInput, Point
from utils import dist

class NodeType(enum.Enum):
    BS = 0
    SN = 1
    RN = 2
    TG = 3

class Node():
    """Node.
    """

    def __init__(self, position, _id, _type=NodeType.SN, is_active=True, forwarding=False):
        self.position = position
        self.id = _id
        self.adj = list()
        self.is_active = is_active
        self.type = _type
        self.forwarding = forwarding

    def __repr__(self):
        return str((self.id, self.position))

    def __str__(self):
        return str((self.id, self.position))

class Sensor(Node):
    """Sensor.
    """

    def __init__(self, position, battery_cap, _id, is_active=True, forwarding=True):
        self.battery_cap = battery_cap
        self.cur_energy = battery_cap
        self.ecr = None
        super(Sensor, self).__init__(position, _id, NodeType.SN, is_active, forwarding)

    def reset(self):
        """reset.
        """
        self.cur_energy = self.battery_cap
        self.activate()

    def deactivate(self):
        """deactivate.
        """
        self.is_active = False

    def activate(self):
        """activate.
        """
        self.is_active = True

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

    def reset(self):
        self.cur_position = self.depot
        self.cur_energy = self.battery_cap

    def move(self, target: Point, t: int):
        pass

class WRSNNetwork():
    """WRSNNetwork.
    """

    def __init__(self, inp: NetworkInput):
        self.num_sensors = inp.num_sensors
        self.num_targets = inp.num_targets
        self.num_charging_points = inp.num_charging_points
        self.num_nodes = self.num_sensors + self.num_targets + 1

        self.sink = Node(inp.sink, 0, NodeType.BS)
        self.sensors = [Sensor(sn, wp.E_s, i) for i, sn in enumerate(inp.sensors, 1)]
        self.targets = [Node(tg, i, NodeType.TG) 
                        for i, tg in enumerate(inp.targets, self.num_sensors+1)]
        self.nodes = list([self.sink, *self.sensors, *self.targets])

        self.depot = inp.depot
        self.charging_points = inp.charging_points
        self.mc = MobileCharger(inp.depot, wp.E_mc, wp.v_mc, wp.e_move, wp.e_mc)
        self.status = {i: False for i in range(self.num_sensors+1, self.num_nodes)}

        self.network_lifetime = 0

        self.__build_adjacency()
        print(self.nodes[0].adj)

    def __build_adjacency(self):
        """__build_adjacency.
        """
        for node in self.nodes:
            for other in self.nodes:
                if node is not other and dist(node.position, other.position) <= wp.r_c:
                    node.adj.append(other)

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

        return trace

    def __estimate_ecr(self):
        """__estimate energy consumption rate.
        """
        pass

    def reset(self):
        """reset.
        """
        for sn in self.sensors:
            sn.reset()
        self.mc.reset()
        self.network_lifetime = 0

    def simulate_communication(self):
        trace = self.__estimate_topology()
        pass

    def t_step(self, t: int):
        return self.__estimate_topology()

if __name__ == '__main__':
    inp = NetworkInput.from_file('data/test/NIn1.json')
    net = WRSNNetwork(inp)
    print(net.t_step(1))
