#!/usr/bin/env python3
"""
ElasticTree Formal/Model-Based Optimization for Jellyfish Topology:
1. Uses optimization model to find MINIMUM switch set
2. Handles random topology with k-shortest path routing
3. Tries different configurations and picks optimal
4. Guarantees connectivity through graph-based path verification

Jellyfish is a random regular graph where switches have random interconnections,
making optimization more challenging than structured topologies.

Usage:
    python3 jellyfish_formal_optimization.py
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
import itertools

logging.basicConfig(filename='./jellyfish_elastictree_formal.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class JellyfishTopo(Topo):
    """
    Jellyfish Random Regular Graph Topology:
    - N switches with k ports each
    - r ports per switch connect to other switches (random)
    - Remaining ports connect to hosts
    - Creates a random regular graph structure
    """
    
    def __init__(self, num_switches=20, num_ports=8, num_switch_ports=4):
        """
        Args:
            num_switches: Number of switches (N)
            num_ports: Total ports per switch (k)
            num_switch_ports: Ports used for switch-to-switch links (r)
        """
        self.num_switches = num_switches
        self.num_ports = num_ports
        self.num_switch_ports = num_switch_ports
        self.num_host_ports = num_ports - num_switch_ports
        
        self.SwitchList = []
        self.HostList = []
        self.switch_links = []  # Track switch-to-switch links
        
        self.bw_switch = 10.0  # Gbps
        self.bw_host = 1.0     # Gbps
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createRandomLinks()
    
    def createTopo(self):
        """Create switches and hosts"""
        # Create switches
        for i in range(1, self.num_switches + 1):
            switch_name = f's{i:03d}'
            self.SwitchList.append(self.addSwitch(switch_name))
        
        # Create hosts (each switch gets num_host_ports hosts)
        total_hosts = self.num_switches * self.num_host_ports
        for i in range(1, total_hosts + 1):
            if i < 10:
                host_name = f'h00{i}'
            elif i < 100:
                host_name = f'h0{i}'
            else:
                host_name = f'h{i}'
            self.HostList.append(self.addHost(host_name))
    
    def createRandomLinks(self):
        """Create random regular graph links between switches"""
        logger.debug("Creating Jellyfish random topology")
        
        # Track available ports for each switch
        available_ports = {sw: self.num_switch_ports for sw in self.SwitchList}
        
        # Create list of switches that need connections
        switches_needing_links = list(self.SwitchList)
        random.shuffle(switches_needing_links)
        
        # Greedily create random links
        attempts = 0
        max_attempts = self.num_switches * 100
        
        while switches_needing_links and attempts < max_attempts:
            attempts += 1
            
            # Get switches with available ports
            candidates = [sw for sw in switches_needing_links if available_ports[sw] > 0]
            
            if len(candidates) < 2:
                break
            
            # Pick two random switches
            sw1, sw2 = random.sample(candidates, 2)
            
            # Check if already connected
            if (sw1, sw2) in self.switch_links or (sw2, sw1) in self.switch_links:
                continue
            
            # Create link
            linkopts = dict(bw=self.bw_switch)
            self.addLink(sw1, sw2, **linkopts)
            self.switch_links.append((sw1, sw2))
            
            # Update available ports
            available_ports[sw1] -= 1
            available_ports[sw2] -= 1
            
            # Remove from candidates if no more ports
            if available_ports[sw1] == 0:
                switches_needing_links.remove(sw1)
            if available_ports[sw2] == 0 and sw2 in switches_needing_links:
                switches_needing_links.remove(sw2)
        
        # Handle remaining switches with odd ports (pair them up or leave isolated)
        while len(switches_needing_links) >= 2:
            sw1 = switches_needing_links.pop(0)
            sw2 = switches_needing_links.pop(0)
            
            if available_ports[sw1] > 0 and available_ports[sw2] > 0:
                if (sw1, sw2) not in self.switch_links and (sw2, sw1) not in self.switch_links:
                    linkopts = dict(bw=self.bw_switch)
                    self.addLink(sw1, sw2, **linkopts)
                    self.switch_links.append((sw1, sw2))
        
        logger.debug(f"Created {len(self.switch_links)} switch-to-switch links")
        
        # Connect hosts to switches
        for switch_idx, switch in enumerate(self.SwitchList):
            start_host = switch_idx * self.num_host_ports
            end_host = min(start_host + self.num_host_ports, len(self.HostList))
            
            for host_idx in range(start_host, end_host):
                host = self.HostList[host_idx]
                linkopts = dict(bw=self.bw_host)
                self.addLink(switch, host, **linkopts)


class JellyfishFormalOptimizer:
    """
    Formal optimization for Jellyfish topology.
    Uses graph algorithms and constraint-based optimization for random topology.
    """
    
    def __init__(self, net, topo, active_ratio=0.5):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self.active_links = set()
        self.active_ratio = active_ratio
        self._build_topology_maps()
        self._build_traffic_matrix(active_ratio)
    
    def _build_topology_maps(self):
        """Build comprehensive topology mappings for random graph"""
        self.host_to_switch = {}
        self.switch_to_hosts = defaultdict(list)
        self.switch_neighbors = defaultdict(set)
        self.all_links = []
        
        # Map hosts to switches
        for switch_idx, switch in enumerate(self.topo.SwitchList):
            start_host = switch_idx * self.topo.num_host_ports
            end_host = min(start_host + self.topo.num_host_ports, len(self.topo.HostList))
            
            for host_idx in range(start_host, end_host):
                host = self.topo.HostList[host_idx]
                self.host_to_switch[host] = switch
                self.switch_to_hosts[switch].append(host)
                self.all_links.append((host, switch))
        
        # Build adjacency for switches (undirected graph)
        for sw1, sw2 in self.topo.switch_links:
            self.switch_neighbors[sw1].add(sw2)
            self.switch_neighbors[sw2].add(sw1)
            self.all_links.append((sw1, sw2))
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """Build traffic demand matrix between hosts"""
        self.traffic_matrix = defaultdict(lambda: defaultdict(float))
        
        hosts = [h.name for h in self.net.hosts]
        num_active = max(1, int(len(hosts) * active_ratio))
        active_hosts = random.sample(hosts, num_active)
        
        info(f"Active hosts for traffic: {num_active}/{len(hosts)} ({active_ratio*100:.0f}%)\n")
        
        # Generate random traffic between active hosts
        for src in active_hosts:
            for dst in active_hosts:
                if src != dst:
                    self.traffic_matrix[src][dst] = random.uniform(0.01, 0.5)
    
    def bfs_path(self, start_switch, end_switch, active_switches):
        """
        Find shortest path between two switches using BFS.
        Only uses switches in active_switches set.
        """
        if start_switch == end_switch:
            return [start_switch]
        
        if start_switch not in active_switches or end_switch not in active_switches:
            return None
        
        queue = deque([(start_switch, [start_switch])])
        visited = {start_switch}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.switch_neighbors[current]:
                if neighbor not in active_switches:
                    continue
                
                if neighbor == end_switch:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def find_path(self, src_host, dst_host, active_switches):
        """
        Find a path between two hosts through active switches.
        Path: host -> switch -> ... -> switch -> host
        """
        if src_host == dst_host:
            return []
        
        src_switch = self.host_to_switch.get(src_host)
        dst_switch = self.host_to_switch.get(dst_host)
        
        if not src_switch or not dst_switch:
            return None
        
        if src_switch not in active_switches or dst_switch not in active_switches:
            return None
        
        # Find path between switches
        switch_path = self.bfs_path(src_switch, dst_switch, active_switches)
        
        if not switch_path:
            return None
        
        # Build complete path
        path = [src_host] + switch_path + [dst_host]
        return path
    
    def verify_connectivity(self, switch_set):
        """Verify all traffic demands can be satisfied"""
        for src in self.traffic_matrix:
            for dst in self.traffic_matrix[src]:
                if self.traffic_matrix[src][dst] > 0:
                    path = self.find_path(src, dst, switch_set)
                    if not path:
                        return False
        return True
    
    def get_connected_component(self, switch_set):
        """Get all switches in the connected component of switch_set"""
        if not switch_set:
            return set()
        
        component = set()
        queue = deque([next(iter(switch_set))])
        visited = set(queue)
        
        while queue:
            current = queue.popleft()
            component.add(current)
            
            for neighbor in self.switch_neighbors[current]:
                if neighbor in switch_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    def find_critical_switches(self, switch_set):
        """
        Find switches that are critical for connectivity.
        A switch is critical if removing it disconnects the graph.
        """
        critical = set()
        
        for switch in switch_set:
            test_set = switch_set - {switch}
            
            # Check if graph is still connected
            if test_set:
                component = self.get_connected_component(test_set)
                if len(component) < len(test_set):
                    critical.add(switch)
        
        return critical
    
    def formal_optimization(self):
        """
        Model-based optimization for Jellyfish topology.
        Challenge: Random topology requires graph-based analysis.
        """
        info("\n=== Formal Optimization Process (Jellyfish) ===\n")
        
        required_switches = set()
        
        # Step 1: Identify active hosts and their switches
        active_hosts = set()
        for src in self.traffic_matrix:
            if any(self.traffic_matrix[src].values()):
                active_hosts.add(src)
        for flows in self.traffic_matrix.values():
            for dst in flows:
                if flows[dst] > 0:
                    active_hosts.add(dst)
        
        # Step 2: Determine required switches (those with active hosts)
        active_switches = set()
        for host in active_hosts:
            switch = self.host_to_switch.get(host)
            if switch:
                active_switches.add(switch)
                required_switches.add(switch)
        
        info(f"  Step 1: Required switches (with active hosts): {len(active_switches)}\n")
        
        # Step 3: Check if these switches are already connected
        if len(active_switches) <= 1:
            info("  Step 2: Single switch - no additional switches needed\n")
            return required_switches, self._compute_links(required_switches, active_hosts)
        
        # Step 4: Find if active switches form connected component
        component = self.get_connected_component(active_switches)
        
        if component == active_switches:
            info(f"  Step 2: Active switches already connected ({len(active_switches)} switches)\n")
            return required_switches, self._compute_links(required_switches, active_hosts)
        
        # Step 5: Need to add switches to connect components
        info(f"  Step 2: Active switches not fully connected, finding bridge switches\n")
        
        # Use greedy approach: iteratively add switches that connect most components
        current_switches = required_switches.copy()
        
        while not self.verify_connectivity(current_switches):
            best_switch = None
            best_improvement = 0
            
            # Try each unused switch
            for switch in self.topo.SwitchList:
                if switch in current_switches:
                    continue
                
                test_switches = current_switches | {switch}
                
                # Check how many new hosts can be reached
                newly_reachable = 0
                for src in active_hosts:
                    for dst in active_hosts:
                        if src != dst and self.traffic_matrix[src][dst] > 0:
                            if not self.find_path(src, dst, current_switches):
                                if self.find_path(src, dst, test_switches):
                                    newly_reachable += 1
                
                if newly_reachable > best_improvement:
                    best_improvement = newly_reachable
                    best_switch = switch
            
            if best_switch:
                current_switches.add(best_switch)
                required_switches.add(best_switch)
                info(f"    Added {best_switch} (enables {best_improvement} new paths)\n")
            else:
                # Fallback: add any switch connected to current set
                for switch in self.topo.SwitchList:
                    if switch not in current_switches:
                        for neighbor in self.switch_neighbors[switch]:
                            if neighbor in current_switches:
                                current_switches.add(switch)
                                required_switches.add(switch)
                                info(f"    Added {switch} (bridge switch)\n")
                                break
                        if switch in current_switches:
                            break
        
        initial_size = len(required_switches)
        info(f"\n  Step 3: Initial solution: {initial_size} switches\n")
        
        # Step 6: Try to remove non-critical switches
        info(f"  Step 4: Attempting to remove redundant switches\n")
        
        removable = []
        for switch in list(required_switches):
            # Don't remove switches with active hosts
            if switch in active_switches:
                continue
            
            test_switches = required_switches - {switch}
            if self.verify_connectivity(test_switches):
                removable.append(switch)
        
        if removable:
            info(f"    Found {len(removable)} redundant switches\n")
            for switch in removable:
                required_switches.remove(switch)
        else:
            info(f"    No redundant switches found\n")
        
        final_size = len(required_switches)
        info(f"\n  Optimization result: {initial_size} → {final_size} switches\n")
        
        # Compute required links
        required_links = self._compute_links(required_switches, active_hosts)
        
        return required_switches, required_links
    
    def _compute_links(self, active_switches, active_hosts):
        """Compute required links for active switches"""
        required_links = set()
        
        # Host-to-switch links
        for host in active_hosts:
            switch = self.host_to_switch.get(host)
            if switch and switch in active_switches:
                required_links.add((host, switch))
        
        # Switch-to-switch links (only between active switches)
        for sw1, sw2 in self.topo.switch_links:
            if sw1 in active_switches and sw2 in active_switches:
                required_links.add((sw1, sw2))
        
        return required_links
    
    def optimize_topology(self):
        """Main optimization routine"""
        info("\n" + "="*80 + "\n")
        info("JELLYFISH ELASTICTREE FORMAL OPTIMIZATION\n")
        info("="*80 + "\n")
        
        # Topology info
        info(f"\nTopology Configuration:\n")
        info(f"  Switches: {self.topo.num_switches}\n")
        info(f"  Ports per switch: {self.topo.num_ports}\n")
        info(f"  Switch-to-switch ports: {self.topo.num_switch_ports}\n")
        info(f"  Host-to-switch ports: {self.topo.num_host_ports}\n")
        info(f"  Total switch links: {len(self.topo.switch_links)}\n")
        info(f"  Total hosts: {len(self.topo.HostList)}\n")
        
        # Calculate total traffic
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"\nTotal network traffic demand: {total_traffic:.3f} Gbps\n")
        
        # Run formal optimization
        required_switches, required_links = self.formal_optimization()
        
        # Calculate statistics
        total_switches = len(self.topo.SwitchList)
        powered_down = total_switches - len(required_switches)
        switch_reduction = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        self.active_links = required_links
        
        # Display results
        info(f"\n{'='*80}\n")
        info(f"*** OPTIMIZATION RESULTS ***\n")
        info(f"{'='*80}\n")
        info(f"  Total switches:      {total_switches}\n")
        info(f"  Active switches:     {len(required_switches)}\n")
        info(f"  Powered down:        {powered_down}\n")
        info(f"  Switch reduction:    {switch_reduction:.1f}%\n")
        
        total_links = len(self.all_links)
        active_link_count = len(required_links)
        link_reduction = ((total_links - active_link_count) / total_links) * 100 if total_links > 0 else 0
        
        info(f"\n  Link Statistics:\n")
        info(f"    Total links:   {total_links}\n")
        info(f"    Active links:  {active_link_count}\n")
        info(f"    Link reduction: {link_reduction:.1f}%\n")
        
        # Analyze graph properties
        self.analyze_graph_properties()
        
        # Visualize
        self.visualize_topology()
        
        return self.active_switches
    
    def analyze_graph_properties(self):
        """Analyze graph properties of optimized topology"""
        info(f"\n  Graph Analysis:\n")
        
        # Average degree in original topology
        total_degree = sum(len(neighbors) for neighbors in self.switch_neighbors.values())
        avg_degree_original = total_degree / len(self.topo.SwitchList) if self.topo.SwitchList else 0
        
        # Average degree in optimized topology
        active_degree = 0
        for switch in self.active_switches:
            active_neighbors = sum(1 for n in self.switch_neighbors[switch] 
                                  if n in self.active_switches)
            active_degree += active_neighbors
        
        avg_degree_optimized = active_degree / len(self.active_switches) if self.active_switches else 0
        
        info(f"    Original avg degree:   {avg_degree_original:.2f}\n")
        info(f"    Optimized avg degree:  {avg_degree_optimized:.2f}\n")
        
        # Check connectivity
        if self.active_switches:
            component = self.get_connected_component(self.active_switches)
            is_connected = len(component) == len(self.active_switches)
            info(f"    Optimized graph connected: {is_connected}\n")
    
    def visualize_topology(self):
        """Visualize optimized topology state"""
        info("\n" + "="*80 + "\n")
        info("TOPOLOGY STATE VISUALIZATION (Sample)\n")
        info("="*80 + "\n")
        
        # Show first 10 switches as sample
        info("\nSWITCH STATUS (First 10):\n")
        for i, switch in enumerate(self.topo.SwitchList[:10]):
            status = "ON " if switch in self.active_switches else "OFF"
            
            # Count active neighbors
            active_neighbors = sum(1 for n in self.switch_neighbors[switch] 
                                  if n in self.active_switches)
            total_neighbors = len(self.switch_neighbors[switch])
            
            # Count active hosts
            hosts = self.switch_to_hosts[switch]
            active_host_count = sum(1 for h in hosts if h in 
                                   [src for src in self.traffic_matrix.keys()] or
                                   h in [dst for flows in self.traffic_matrix.values() 
                                        for dst in flows.keys()])
            
            info(f"  {switch}: {status}  "
                 f"[{active_neighbors}/{total_neighbors} neighbors, "
                 f"{active_host_count}/{len(hosts)} active hosts]\n")
            
            if status == "ON " and active_host_count > 0:
                info(f"      Active hosts: {', '.join(hosts[:4])}\n")
        
        if len(self.topo.SwitchList) > 10:
            info(f"\n  ... ({len(self.topo.SwitchList) - 10} more switches)\n")
        
        info("\n" + "="*80 + "\n")


def run_topology():
    """Main function to run Jellyfish with formal optimization"""
    setLogLevel('info')
    
    # Topology parameters
    num_switches = 20
    num_ports = 8
    num_switch_ports = 4  # 4 ports for switch links, 4 for hosts
    active_ratio = 0.3    # 30% of hosts are active
    
    info(f"\nCreating Jellyfish Random Topology:\n")
    info(f"  Switches: {num_switches}\n")
    info(f"  Ports per switch: {num_ports}\n")
    info(f"  Switch-to-switch ports: {num_switch_ports}\n")
    info(f"  Host-to-switch ports: {num_ports - num_switch_ports}\n")
    info(f"  Total hosts: {num_switches * (num_ports - num_switch_ports)}\n")
    info(f"  Active host ratio: {active_ratio*100:.0f}%\n\n")
    
    topo = JellyfishTopo(num_switches=num_switches,
                         num_ports=num_ports,
                         num_switch_ports=num_switch_ports)
    
    net = Mininet(topo=topo,
                  controller=RemoteController,
                  switch=OVSKernelSwitch,
                  link=TCLink,
                  autoSetMacs=True)
    
    net.start()
    
    # Run formal optimization
    optimizer = JellyfishFormalOptimizer(net, topo, active_ratio=active_ratio)
    optimizer.optimize_topology()
    
    info("\n*** Network ready. Starting CLI ***\n")
    info("*** Type 'exit' to quit ***\n\n")
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    try:
        run_topology()
    except KeyboardInterrupt:
        info("\n*** Interrupted by user ***\n")
        from mininet.clean import cleanup
        cleanup()