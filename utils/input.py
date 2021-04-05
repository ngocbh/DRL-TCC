from collections import namedtuple, deque, defaultdict

import json
from utils.utils import dist
from utils.utils import Point

class NetworkInput():

    def __init__(self, W=500, H=500, num_sensors=0, num_relays=0, num_targets=0,
                 num_charging_points=0, sink=None, depot=None, sensors=None, 
                 relays=None, targets=None, charging_points=None, r_c=25, r_s=25):
        self.W = W
        self.H = H
        self.num_sensors = num_sensors
        self.num_relays = num_relays
        self.num_targets = num_targets


        self.sink = sink
        self.depot = depot
        self.sensors = sensors
        self.relays = relays
        self.targets = targets

        if charging_points is None:
            self.num_charging_points = num_sensors
            self.charging_points = sensors
        else:
            self.num_charging_points = num_charging_points
            self.charging_points = charging_points

        self.r_c = r_c
        self.r_s = r_s

    def __hash__(self):
        return hash((self.W, self.H, self.num_relays, self.num_sensors, self.r_c, self.r_s,
                     tuple(self.relays), tuple(self.sensors), tuple(self.targets), 
                     tuple(self.charging_points)))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_file(cls, filepath):
        data = json.load(open(filepath, mode='r'))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data):
        W = data['W']
        H = data['H']

        sink = Point(**data['sink'])
        depot = Point(**data['depot'])
        r_c = data['communication_range']

        if 'num_of_sensors' in data and 'sensors' in data:
            num_sensors = data['num_of_sensors']
            sensors = [Point(**e) for e in data['sensors']]
        else:
            num_sensors = 0
            sensors = None

        if 'num_of_relays' in data and 'relays' in data:
            num_relays = data['num_of_relays']
            sensors = [Point(**e) for e in data['relays']]
        else:
            num_relays = 0
            relays = None

        if 'num_of_targets' in data and 'targets' in data:
            num_targets = data['num_of_targets']
            targets = [Point(**e) for e in data['targets']]
        else:
            num_targets = 0
            targets = None

        if 'num_of_charging_points' in data and 'charging_points' in data \
                and data['charging_points']:
            num_charging_points = data['num_of_charging_points']
            charging_points = [Point(**e) for e in data['charging_points']]
        else:
            num_charging_points = data['num_of_sensors']
            charging_points = [Point(**e) for e in data['sensors']]

        return cls(W, H, num_sensors, num_relays, num_targets,
                   num_charging_points, sink, depot, sensors, relays,
                   targets, charging_points, r_c)

    def freeze(self):
        self.sensors = tuple(self.sensors)
        self.targets = tuple(self.targets)
        self.charging_points = tuple(self.charging_points)
        self.relays = tuple(self.relays)

    def to_dict(self):
        return {
            'W': self.W, 'H': self.H,
            'num_of_relays': self.num_relays,
            'num_of_sensors': self.num_sensors,
            'num_of_charging_points': self.num_charging_points,
            'num_of_targets': self.num_targets,
            'relays': list(map(lambda x: x._asdict(), self.relays)),
            'sensors': list(map(lambda x: x._asdict(), self.sensors)),
            'targets': list(map(lambda x: x._asdict(), self.targets)),
            'charging_points': list(map(lambda x: x._asdict(), self.charging_points)),
            'sink': self.sink._asdict(),
            'depot': self.depot._asdict(),
            'communication_range': self.r_c,
            'sensing_range': self.r_s
        }

    def to_file(self, file_path):
        d = self.to_dict()
        with open(file_path, "wt") as f:
            fstr = json.dumps(d, indent=4)
            f.write(fstr)

    def is_connected(self):
        visited = defaultdict(lambda : False)
        queue = deque()
        queue.append(self.sink)
        
        while queue:
            u = queue.popleft()
            visited[u] = True

            for sn in self.sensors:
                if not visited[sn] and dist(u, sn) <=  self.r_c:
                    queue.append(sn)

        if any(not visited[sn] for sn in self.sensors):
            return False

        for tg in self.targets:
            if all(dist(tg, sn) > self.r_c for sn in self.sensors):
                return False

        return True

