from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import Controller, OVSSwitch, OVSKernelSwitch, Host, RemoteController
from mininet.link import TCLink
import pickle
import random

class RandomizedSubstrateNetwork(Topo):
    def build(self, num_servers=10, num_switches=5, num_links=15):
        servers = []
        for i in range(num_servers):
            cpu = random.randint(50, 200)       # CPU (50-200 units)
            memory = random.randint(1000, 8000)  # Memory (1GB-8GB)
            storage = random.randint(10000, 50000)  # Storage (10GB-50GB)
            server = self.addHost(f'server{i}', cpu=cpu, memory=memory, storage=storage)
            servers.append(server)

        switches = []
        for i in range(num_switches):
            switch = self.addSwitch(f'switch{i}')
            switches.append(switch)

        for i in range(num_links):
            server = random.choice(servers)
            switch = random.choice(switches)
            bandwidth = random.randint(100, 1000)  # Bandwidth (100-1000 Mbps)
            delay = f"{random.randint(1, 10)}ms"  # Random delay (1-10 ms)
            self.addLink(server, switch, bw=bandwidth, delay=delay)

        for i in range(len(switches) - 1):
            self.addLink(switches[i], switches[i + 1], bw=random.randint(500, 1000))


def create_randomized_mininet(filename):
    topo = RandomizedSubstrateNetwork()
    net = Mininet(controller=None, link=TCLink, switch=OVSKernelSwitch)
    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           ip='127.0.0.1',
                           port=6633)
    net.start()

    substrate_info = {
        "servers": [
            {
                "id": host.name,
                "cpu": host.params.get("cpu", "N/A"),
                "memory": host.params.get("memory", "N/A"),
                "storage": host.params.get("storage", "N/A")
            }
            for host in net.hosts
        ],
        "switches": [switch.name for switch in net.switches],
        "links": [
            {
                "src": link.intf1.node.name,
                "dst": link.intf2.node.name,
                "bandwidth": link.intf1.params.get('bw', 'N/A'),
                "delay": link.intf1.params.get('delay', 'N/A')
            }
            for link in net.links
        ]
    }

    with open(filename, 'wb') as f:
        pickle.dump(substrate_info, f)

    print(f"Randomized Mininet substrate network saved to {filename}")
    net.stop()

# Generate and save a randomized substrate network
create_randomized_mininet("substrate_network.pkl")