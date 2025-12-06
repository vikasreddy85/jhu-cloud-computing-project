from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import random
from collections import defaultdict, deque

logger = logging.getLogger(__name__)
class Jellyfish(Topo):
    def __init__(self, num_switches=20, num_ports=4, num_hosts_per_switch=2):
        self.num_switches = num_switches
        self.num_ports = num_ports
        self.num_hosts_per_switch = num_hosts_per_switch        
        self.switch_to_switch_ports = num_ports - num_hosts_per_switch
        
        self.bw_switch = 1.0
        self.bw_host = 1.0
        
        self.SwitchList = []
        self.HostList = []
        self.switch_links = defaultdict(set)
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createRandomLinks()
    
    def createTopo(self):
        """Create switches and hosts"""
        for i in range(1, self.num_switches + 1):
            switch_name = f's{i:03d}'
            self.SwitchList.append(self.addSwitch(switch_name))        
        total_hosts = self.num_switches * self.num_hosts_per_switch
        for i in range(1, total_hosts + 1):
            host_name = f'h{i:03d}'
            self.HostList.append(self.addHost(host_name))
    
    def createRandomLinks(self):
        available_ports = {s: self.switch_to_switch_ports for s in self.SwitchList}
        
        switches_with_ports = list(self.SwitchList)
        random.shuffle(switches_with_ports)
                
        while len(switches_with_ports) >= 2:
            s1 = switches_with_ports[0]
            switches_with_ports.remove(s1)
            
            if not switches_with_ports:
                break
            
            s2 = random.choice(switches_with_ports)
            
            if (available_ports[s1] > 0 and available_ports[s2] > 0 and 
                s2 not in self.switch_links[s1]):
                
                linkopts = dict(bw=self.bw_switch)
                self.addLink(s1, s2, **linkopts)
                
                self.switch_links[s1].add(s2)
                self.switch_links[s2].add(s1)
                
                available_ports[s1] -= 1
                available_ports[s2] -= 1
                
                if available_ports[s1] == 0:
                    switches_with_ports = [s for s in switches_with_ports if s != s1]
                if available_ports[s2] == 0:
                    switches_with_ports.remove(s2)
        
        for switch_idx, switch in enumerate(self.SwitchList):
            start_host_idx = switch_idx * self.num_hosts_per_switch
            for i in range(self.num_hosts_per_switch):
                host_idx = start_host_idx + i
                if host_idx < len(self.HostList):
                    linkopts = dict(bw=self.bw_host)
                    self.addLink(switch, self.HostList[host_idx], **linkopts)
        
        total_links = sum(len(neighbors) for neighbors in self.switch_links.values()) // 2
        info(f"Created {total_links} switch-to-switch links\n")
        info(f"Connected {len(self.HostList)} hosts\n")


class GreedyBinPackingOptimizer:
    def __init__(self, net, topo):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self._build_topology_maps()
    
    def _build_topology_maps(self):
        """Build mappings between topology elements"""
        self.host_to_switch = {}
        self.switch_to_hosts = defaultdict(list)
        
        for switch_idx, switch in enumerate(self.topo.SwitchList):
            start_host_idx = switch_idx * self.topo.num_hosts_per_switch
            for i in range(self.topo.num_hosts_per_switch):
                host_idx = start_host_idx + i
                if host_idx < len(self.topo.HostList):
                    host = self.topo.HostList[host_idx]
                    self.host_to_switch[host] = switch
                    self.switch_to_hosts[switch].append(host)
    
    def bfs_shortest_paths(self, active_switches):
        """
        Compute shortest paths between all pairs of active switches using BFS
        Returns: dict mapping (src, dst) -> path length
        """
        paths = {}
        
        for src in active_switches:
            # BFS from src
            queue = deque([(src, 0)])
            visited = {src}
            distances = {src: 0}
            
            while queue:
                current, dist = queue.popleft()
                
                for neighbor in self.topo.switch_links[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
            
            for dst in active_switches:
                if dst != src:
                    paths[(src, dst)] = distances.get(dst, float('inf'))
        
        return paths
    
    def find_min_switches_for_connectivity(self, required_switches):
        """
        Find minimum set of additional switches needed to ensure connectivity
        between all required switches using greedy approach
        """
        if len(required_switches) <= 1:
            return set()
        
        additional = set()
        current_active = set(required_switches)
        
        max_iterations = 10
        for _ in range(max_iterations):
            paths = self.bfs_shortest_paths(current_active)
            
            disconnected_pairs = []
            for src in required_switches:
                for dst in required_switches:
                    if src != dst and paths.get((src, dst), float('inf')) == float('inf'):
                        disconnected_pairs.append((src, dst))
            
            if not disconnected_pairs:
                break
            
            candidate_switches = set(self.topo.SwitchList) - current_active
            best_switch = None
            best_improvement = 0
            
            for candidate in candidate_switches:
                test_active = current_active | {candidate}
                test_paths = self.bfs_shortest_paths(test_active)
                
                improvement = sum(1 for src, dst in disconnected_pairs 
                                if test_paths.get((src, dst), float('inf')) < float('inf'))
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_switch = candidate
            
            if best_switch:
                additional.add(best_switch)
                current_active.add(best_switch)
            else:
                break
        
        return additional
    
    def get_host_traffic(self):
        """Simulate or measure host traffic"""
        traffic = {}
        for host in self.net.hosts:
            traffic[host.name] = random.uniform(0.01, 0.5)
        return traffic
    
    def greedy_bin_packing(self, host_traffic):
        required_switches = set()
        
        if not host_traffic:
            return required_switches, {}
        
        sorted_hosts = sorted(host_traffic.items(), key=lambda x: x[1], reverse=True)
        
        switch_utilization = defaultdict(float)
        switch_capacity = self.topo.bw_host * self.topo.num_hosts_per_switch
        for host_name, traffic in sorted_hosts:
            assigned = False
            host = next((h for h in self.topo.HostList if h == host_name), None)
            
            if not host or host not in self.host_to_switch:
                continue
            
            original_switch = self.host_to_switch[host]
            
            for switch in self.topo.SwitchList:
                if switch in required_switches:
                    if switch_utilization[switch] + traffic <= switch_capacity:
                        switch_utilization[switch] += traffic
                        assigned = True
                        info(f"  {host_name} ({traffic:.3f} Gbps) -> {switch} "
                             f"(util: {switch_utilization[switch]:.3f}/{switch_capacity:.2f})\n")
                        break
            
            if not assigned:
                if original_switch not in required_switches:
                    switch_utilization[original_switch] = traffic
                    required_switches.add(original_switch)
                    info(f"  {host_name} ({traffic:.3f} Gbps) -> {original_switch} [NEW] "
                         f"(util: {switch_utilization[original_switch]:.3f}/{switch_capacity:.2f})\n")
                else:
                    for switch in self.topo.SwitchList:
                        if switch not in required_switches:
                            switch_utilization[switch] = traffic
                            required_switches.add(switch)
                            info(f"  {host_name} ({traffic:.3f} Gbps) -> {switch} [NEW] "
                                 f"(util: {switch_utilization[switch]:.3f}/{switch_capacity:.2f})\n")
                            break
        
        connectivity_switches = self.find_min_switches_for_connectivity(required_switches)
        
        if connectivity_switches:
            info(f"Added {len(connectivity_switches)} switches for connectivity: "
                 f"{sorted(connectivity_switches)}\n")
            required_switches.update(connectivity_switches)
        else:
            info("All required switches are already connected\n")
        
        return required_switches, switch_utilization
    
    def visualize_topology(self, switch_utilization):
        """Visualize the topology with utilization information"""
        switch_capacity = self.topo.bw_host * self.topo.num_hosts_per_switch
        
        info("\nSWITCH STATUS:\n")
        for switch in self.topo.SwitchList:
            status = "ON " if switch in self.active_switches else "OFF"
            
            num_neighbors = len(self.topo.switch_links[switch])
            active_neighbors = sum(1 for n in self.topo.switch_links[switch] 
                                  if n in self.active_switches)
            
            if switch in self.active_switches and switch in switch_utilization:
                util = switch_utilization[switch]
                util_pct = (util / switch_capacity) * 100 if switch_capacity > 0 else 0
                info(f"  {switch}: {status} [Util: {util:.3f}/{switch_capacity:.2f} Gbps = {util_pct:.1f}%] "
                     f"[Links: {active_neighbors}/{num_neighbors} active]\n")
            else:
                info(f"  {switch}: {status} [Links: {active_neighbors}/{num_neighbors} active]\n")
            
            hosts = self.switch_to_hosts[switch]
            if hosts:
                info(f"         Hosts: {', '.join(hosts)}\n")
        
        info("\n" + "="*80 + "\n\n")
    
    def optimize_topology(self):
        """Main optimization routine"""        
        host_traffic = self.get_host_traffic()
        total_traffic = sum(host_traffic.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        info(f"Active hosts: {len(host_traffic)}\n")
        
        required_switches, switch_utilization = self.greedy_bin_packing(host_traffic)        
        total_switches = len(self.topo.SwitchList)
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Power savings:      {power_savings:.1f}%\n")
        
        if required_switches:
            active_utils = [switch_utilization[s] for s in required_switches 
                          if s in switch_utilization]
            if active_utils:
                avg_util = sum(active_utils) / len(active_utils)
                switch_capacity = self.topo.bw_host * self.topo.num_hosts_per_switch
                avg_util_pct = (avg_util / switch_capacity) * 100 if switch_capacity > 0 else 0
                info(f"\n  Average switch utilization: {avg_util:.3f} Gbps ({avg_util_pct:.1f}%)\n")
        
        total_links = sum(len(neighbors) for neighbors in self.topo.switch_links.values()) // 2
        active_links = sum(1 for s1 in required_switches 
                          for s2 in self.topo.switch_links[s1] 
                          if s2 in required_switches) // 2
        info(f"  Active links:       {active_links}/{total_links}\n")
        
        self.visualize_topology(switch_utilization)
        
        return self.active_switches


def run_topology():
    setLogLevel('info')
    
    num_switches = 20
    num_ports = 4
    num_hosts_per_switch = 2
    
    topo = Jellyfish(
        num_switches=num_switches,
        num_ports=num_ports,
        num_hosts_per_switch=num_hosts_per_switch
    )
    
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    net.start()
    
    info(f"  Switches:              {num_switches}\n")
    info(f"  Ports per switch:      {num_ports}\n")
    info(f"  Hosts per switch:      {num_hosts_per_switch}\n")
    info(f"  Total hosts:           {num_switches * num_hosts_per_switch}\n")
    info(f"  Switch-to-switch ports: {num_ports - num_hosts_per_switch}\n")
    
    optimizer = GreedyBinPackingOptimizer(net, topo)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()