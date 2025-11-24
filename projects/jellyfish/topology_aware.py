#!/usr/bin/env python3
"""
ElasticTree Topology-aware for Jellyfish:
1. Uses random regular graph structure for routing-aware decisions
2. Considers switch-level affinity and communication patterns
3. Groups communicating hosts on same or nearby switches when possible
4. Minimizes inter-switch traffic by utilizing locality

Usage:
    python3 jellyfish_topology_aware.py
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import random
from collections import defaultdict, deque

logging.basicConfig(filename='./jellyfish_elastictree.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Jellyfish(Topo):
    def __init__(self, num_switches=20, num_ports=4, num_hosts=80):
        """
        Jellyfish topology
        num_switches: number of switches (N)
        num_ports: ports per switch (k)
        num_hosts: total hosts to connect
        """
        self.num_switches = num_switches
        self.num_ports = num_ports
        self.num_hosts = num_hosts
        
        # Calculate switch-to-switch and host connections
        self.ports_to_switches = num_ports - 1  # Reserve 1 port minimum for hosts
        self.hosts_per_switch = max(1, num_hosts // num_switches)
        
        self.bw_s2s = 1.0  # Switch-to-switch bandwidth
        self.bw_h2s = 1.0  # Host-to-switch bandwidth
        
        self.SwitchList = []
        self.HostList = []
        self.switch_links = defaultdict(list)  # Track switch-to-switch links
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createLinks()
    
    def createTopo(self):
        """Create switches and hosts"""
        # Create switches
        for i in range(1, self.num_switches + 1):
            switch_name = f's{i:03d}'
            self.SwitchList.append(self.addSwitch(switch_name))
        
        # Create hosts
        for i in range(1, self.num_hosts + 1):
            host_name = f'h{i:03d}'
            self.HostList.append(self.addHost(host_name))
    
    def createLinks(self):
        """
        Create Jellyfish random regular graph links
        Uses a randomized algorithm to create a k-regular random graph
        """
        info(f"\n*** Creating Jellyfish topology ***\n")
        info(f"  Switches: {self.num_switches}, Ports: {self.num_ports}\n")
        
        # Create random regular graph for switch-to-switch connections
        self._create_random_regular_graph()
        
        # Connect hosts to switches
        self._connect_hosts_to_switches()
        
        info(f"  Total switch-switch links: {sum(len(links) for links in self.switch_links.values()) // 2}\n")
        info(f"  Total hosts: {self.num_hosts}\n")
    
    def _create_random_regular_graph(self):
        """
        Create a random k-regular graph using the configuration model
        Each switch gets exactly k edges (or as close as possible)
        """
        # Calculate how many ports each switch dedicates to other switches
        ports_for_switches = []
        for i in range(self.num_switches):
            # Reserve at least 1 port per switch for hosts
            hosts_on_switch = min(self.hosts_per_switch, 
                                 self.num_hosts - i * self.hosts_per_switch)
            hosts_on_switch = max(1, hosts_on_switch)
            available = self.num_ports - hosts_on_switch
            ports_for_switches.append(max(1, available))
        
        # Create edge stubs (half-edges)
        stubs = []
        for switch_idx, num_ports in enumerate(ports_for_switches):
            for _ in range(num_ports):
                stubs.append(switch_idx)
        
        # If odd number of stubs, remove one
        if len(stubs) % 2 == 1:
            stubs.pop()
        
        # Shuffle and pair up stubs
        random.shuffle(stubs)
        
        connected_pairs = set()
        i = 0
        max_attempts = len(stubs) * 2
        attempts = 0
        
        while i < len(stubs) - 1 and attempts < max_attempts:
            s1_idx = stubs[i]
            s2_idx = stubs[i + 1]
            
            # Avoid self-loops and duplicate edges
            if s1_idx != s2_idx:
                pair = tuple(sorted([s1_idx, s2_idx]))
                if pair not in connected_pairs:
                    s1 = self.SwitchList[s1_idx]
                    s2 = self.SwitchList[s2_idx]
                    
                    linkopts = dict(bw=self.bw_s2s)
                    self.addLink(s1, s2, **linkopts)
                    
                    self.switch_links[s1].append(s2)
                    self.switch_links[s2].append(s1)
                    connected_pairs.add(pair)
                    
                    i += 2
                else:
                    # Try re-pairing
                    if i + 3 < len(stubs):
                        stubs[i + 1], stubs[i + 3] = stubs[i + 3], stubs[i + 1]
            else:
                # Self-loop detected, re-pair
                if i + 2 < len(stubs):
                    stubs[i + 1], stubs[i + 2] = stubs[i + 2], stubs[i + 1]
            
            attempts += 1
    
    def _connect_hosts_to_switches(self):
        """Connect hosts to switches in a balanced way"""
        for host_idx, host in enumerate(self.HostList):
            switch_idx = host_idx // self.hosts_per_switch
            if switch_idx >= len(self.SwitchList):
                switch_idx = len(self.SwitchList) - 1
            
            switch = self.SwitchList[switch_idx]
            linkopts = dict(bw=self.bw_h2s)
            self.addLink(host, switch, **linkopts)


class ElasticTreeOptimizer:
    def __init__(self, net, topo, active_ratio=0.5):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self.active_ratio = active_ratio
        self._build_topology_maps()
        self._build_traffic_matrix(active_ratio)
    
    def _build_topology_maps(self):
        """Build topology connectivity maps for Jellyfish"""
        self.host_to_switch = {}
        self.switch_to_hosts = defaultdict(list)
        
        # Map hosts to their directly connected switch
        hosts_per_switch = self.topo.hosts_per_switch
        for i, host in enumerate(self.topo.HostList):
            switch_idx = min(i // hosts_per_switch, len(self.topo.SwitchList) - 1)
            switch = self.topo.SwitchList[switch_idx]
            self.host_to_switch[host] = switch
            self.switch_to_hosts[switch].append(host)
        
        # Build switch adjacency (already in topo.switch_links)
        self.switch_neighbors = self.topo.switch_links
        
        # Calculate shortest paths between all switch pairs (for routing awareness)
        self._compute_switch_distances()
    
    def _compute_switch_distances(self):
        """Compute shortest path distances between all switches using BFS"""
        self.switch_distance = {}
        
        for src_switch in self.topo.SwitchList:
            distances = {src_switch: 0}
            queue = deque([src_switch])
            
            while queue:
                current = queue.popleft()
                current_dist = distances[current]
                
                for neighbor in self.switch_neighbors[current]:
                    if neighbor not in distances:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
            
            self.switch_distance[src_switch] = distances
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """Build traffic demand matrix between hosts"""
        self.traffic_matrix = defaultdict(lambda: defaultdict(float))
        
        hosts = [h.name for h in self.net.hosts]
        num_active = max(1, int(len(hosts) * active_ratio))
        active_hosts = random.sample(hosts, num_active)
        
        info(f"\n*** Traffic Generation ***\n")
        info(f"Active hosts: {num_active}/{len(hosts)} ({active_ratio*100:.0f}%)\n")
        info(f"Active: {', '.join(sorted(active_hosts[:10]))}{'...' if len(active_hosts) > 10 else ''}\n")
        
        # Generate traffic with locality bias
        for src in active_hosts:
            for dst in active_hosts:
                if src != dst:
                    # Add some locality: hosts on same switch communicate more
                    src_host = next(h for h in self.net.hosts if h.name == src)
                    dst_host = next(h for h in self.net.hosts if h.name == dst)
                    src_switch = self.host_to_switch.get(src_host.name, None)
                    dst_switch = self.host_to_switch.get(dst_host.name, None)
                    
                    if src_switch == dst_switch:
                        # Same switch: higher traffic
                        traffic = random.uniform(0.1, 0.8)
                    else:
                        # Different switches: lower traffic
                        traffic = random.uniform(0.01, 0.3)
                    
                    self.traffic_matrix[src][dst] = traffic
    
    def build_affinity_groups(self):
        """
        Build affinity groups: clusters of hosts that communicate heavily
        For Jellyfish, we try to group hosts that should be on the same switch
        """
        info(f"\n*** Building Communication Affinity Groups ***\n")
        
        # Calculate total traffic between each host pair
        host_traffic = defaultdict(float)
        for src in self.traffic_matrix:
            for dst in self.traffic_matrix[src]:
                traffic = self.traffic_matrix[src][dst]
                pair = tuple(sorted([src, dst]))
                host_traffic[pair] += traffic
        
        # Get all active hosts
        all_hosts = set()
        for src in self.traffic_matrix:
            all_hosts.add(src)
            for dst in self.traffic_matrix[src]:
                all_hosts.add(dst)
        
        # Sort host pairs by traffic volume
        sorted_pairs = sorted(host_traffic.items(), key=lambda x: x[1], reverse=True)
        
        affinity_groups = []
        assigned_hosts = set()
        hosts_per_switch = self.topo.hosts_per_switch
        
        # Greedily form groups that should be co-located on same switch
        for (h1, h2), traffic in sorted_pairs:
            if h1 in assigned_hosts or h2 in assigned_hosts:
                continue
            
            # Start a new affinity group
            group = {h1, h2}
            assigned_hosts.add(h1)
            assigned_hosts.add(h2)
            
            # Try to add more hosts with high affinity
            for host in all_hosts:
                if host in assigned_hosts or len(group) >= hosts_per_switch:
                    break
                
                # Check if this host has high traffic with group members
                has_affinity = False
                total_affinity = 0
                for member in group:
                    pair = tuple(sorted([host, member]))
                    if pair in host_traffic:
                        total_affinity += host_traffic[pair]
                
                # If average affinity is high enough, add to group
                if total_affinity / len(group) > 0.05:
                    group.add(host)
                    assigned_hosts.add(host)
            
            affinity_groups.append(group)
        
        # Add remaining hosts as singleton groups
        for host in all_hosts:
            if host not in assigned_hosts:
                affinity_groups.append({host})
        
        info(f"Created {len(affinity_groups)} affinity groups:\n")
        for i, group in enumerate(affinity_groups[:10]):
            info(f"  Group {i}: {sorted(group)} (size: {len(group)})\n")
        if len(affinity_groups) > 10:
            info(f"  ... and {len(affinity_groups) - 10} more groups\n")
        
        return affinity_groups
    
    def topology_aware_placement(self, affinity_groups):
        """
        Place affinity groups intelligently on Jellyfish switches
        Try to minimize switch hops between communicating hosts
        """
        required_switches = set()
        
        info(f"\n*** Topology-Aware Placement ***\n")
        
        # Sort groups by size (larger groups first)
        sorted_groups = sorted(affinity_groups, key=lambda g: len(g), reverse=True)
        
        # Track which switches are assigned
        switch_assignment = {}  # frozenset(group) -> switch
        available_switches = list(self.topo.SwitchList)
        
        # Assign each group to a switch
        for group in sorted_groups:
            if not available_switches:
                # Reuse switches if we run out
                available_switches = list(self.topo.SwitchList)
            
            switch = available_switches.pop(0)
            switch_assignment[frozenset(group)] = switch
            required_switches.add(switch)
            
            info(f"  Group {sorted(list(group))[:3]}{'...' if len(group) > 3 else ''} -> {switch}\n")
        
        # Now calculate which additional switches are needed for routing
        # Find switches that are on paths between active switches
        active_switches = set(switch_assignment.values())
        
        # Use connectivity-based approach: add neighbor switches if they help routing
        switches_to_check = list(active_switches)
        checked = set()
        
        while switches_to_check:
            switch = switches_to_check.pop(0)
            if switch in checked:
                continue
            checked.add(switch)
            
            # Check if any neighbors would help connectivity
            for neighbor in self.switch_neighbors[switch]:
                if neighbor in required_switches:
                    continue
                
                # Add neighbor if it connects to multiple active switches
                connections_to_active = sum(
                    1 for n in self.switch_neighbors[neighbor]
                    if n in active_switches
                )
                
                if connections_to_active >= 2:
                    required_switches.add(neighbor)
                    switches_to_check.append(neighbor)
        
        # Calculate traffic statistics
        intra_switch_traffic = 0
        inter_switch_traffic = 0
        
        for src in self.traffic_matrix:
            src_switch = None
            for group, switch in switch_assignment.items():
                if src in group:
                    src_switch = switch
                    break
            
            if not src_switch:
                continue
            
            for dst in self.traffic_matrix[src]:
                dst_switch = None
                for group, switch in switch_assignment.items():
                    if dst in group:
                        dst_switch = switch
                        break
                
                if not dst_switch:
                    continue
                
                traffic = self.traffic_matrix[src][dst]
                if src_switch == dst_switch:
                    intra_switch_traffic += traffic
                else:
                    inter_switch_traffic += traffic
        
        info(f"\n*** Traffic Analysis ***\n")
        info(f"  Active switches: {len(active_switches)}\n")
        info(f"  Additional routing switches: {len(required_switches) - len(active_switches)}\n")
        info(f"  Intra-switch traffic: {intra_switch_traffic:.3f} Gbps\n")
        info(f"  Inter-switch traffic: {inter_switch_traffic:.3f} Gbps\n")
        
        locality = (intra_switch_traffic / (intra_switch_traffic + inter_switch_traffic) * 100 
                   if (intra_switch_traffic + inter_switch_traffic) > 0 else 0)
        info(f"  Traffic locality: {locality:.1f}%\n")
        
        return required_switches
    
    def visualize_topology(self):
        """Display the current topology state"""
        info("\n" + "="*80 + "\n")
        info("TOPOLOGY STATE (JELLYFISH TOPOLOGY-AWARE AFFINITY PLACEMENT)\n")
        info("="*80 + "\n")
        
        info("\nSWITCH STATUS:\n")
        for i, switch in enumerate(self.topo.SwitchList):
            status = "ON " if switch in self.active_switches else "OFF"
            
            # Count active neighbors
            active_neighbors = sum(
                1 for n in self.switch_neighbors[switch]
                if n in self.active_switches
            )
            
            # Get hosts on this switch
            hosts = self.switch_to_hosts.get(switch, [])
            
            info(f"  {switch}: {status} | Neighbors: {active_neighbors}/{len(self.switch_neighbors[switch])} | Hosts: {len(hosts)}\n")
            
            if i < 10 or (status == "ON " and i < 20):
                if hosts:
                    info(f"         Hosts: {', '.join(hosts)}\n")
        
        info("="*80 + "\n\n")
    
    def optimize_topology(self):
        """Main optimization routine"""
        info("\n*** ElasticTree Topology-Aware Optimization for Jellyfish ***\n")
        
        # Calculate total traffic
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        
        # Build affinity groups based on communication patterns
        affinity_groups = self.build_affinity_groups()
        
        # Place groups in topology-aware manner
        required_switches = self.topology_aware_placement(affinity_groups)
        
        total_switches = len(self.topo.SwitchList)
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"\n*** Optimization Results ***\n")
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Power savings:      {power_savings:.1f}%\n")
        
        self.visualize_topology()
        
        return self.active_switches


def run_topology():
    """Main entry point"""
    setLogLevel('info')
    
    # Jellyfish parameters
    num_switches = 20
    num_ports = 4
    num_hosts = 80
    
    topo = Jellyfish(
        num_switches=num_switches,
        num_ports=num_ports,
        num_hosts=num_hosts
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
    
    info(f"\n*** Jellyfish Topology ***\n")
    info(f"  Switches:         {num_switches}\n")
    info(f"  Ports per switch: {num_ports}\n")
    info(f"  Hosts:            {num_hosts}\n")
    info(f"  Hosts per switch: ~{num_hosts // num_switches}\n")
    
    # Run optimization with 50% active hosts
    optimizer = ElasticTreeOptimizer(net, topo, active_ratio=0.5)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()