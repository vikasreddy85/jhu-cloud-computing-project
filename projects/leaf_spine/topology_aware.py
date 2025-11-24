#!/usr/bin/env python3
"""
ElasticTree Topology-aware for Leaf-Spine:
1. Groups communicating hosts under same leaf switches
2. Minimizes inter-leaf traffic
3. Only activates spine switches needed for cross-leaf communication
4. Optimizes based on communication affinity patterns

Usage:
    python3 projects/leafspine/topology_aware.py
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import random
from collections import defaultdict

logging.basicConfig(filename='./leafspine_elastictree.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LeafSpineTopo(Topo):
    """
    Leaf-Spine Topology:
    - num_spines: Number of spine (core) switches
    - num_leaves: Number of leaf (ToR) switches
    - hosts_per_leaf: Number of hosts connected to each leaf
    """
    def __init__(self, num_spines=4, num_leaves=8, hosts_per_leaf=4):
        self.num_spines = num_spines
        self.num_leaves = num_leaves
        self.hosts_per_leaf = hosts_per_leaf
        self.total_hosts = num_leaves * hosts_per_leaf
        
        self.bw_spine_leaf = 10.0
        self.bw_leaf_host = 1.0
        
        self.SpineSwitchList = []
        self.LeafSwitchList = []
        self.HostList = []
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createLinks()
    
    def createTopo(self):
        """Create switches and hosts"""
        for i in range(1, self.num_spines + 1):
            spine = self.addSwitch(f's{i:03d}')
            self.SpineSwitchList.append(spine)
        
        for i in range(1, self.num_leaves + 1):
            leaf = self.addSwitch(f'l{i:03d}')
            self.LeafSwitchList.append(leaf)
        
        for i in range(1, self.total_hosts + 1):
            if i < 10:
                host = self.addHost(f'h00{i}')
            elif i < 100:
                host = self.addHost(f'h0{i}')
            else:
                host = self.addHost(f'h{i}')
            self.HostList.append(host)
    
    def createLinks(self):
        """Create full mesh between spines and leaves, and leaf-to-host links"""
        logger.debug("Creating Spine to Leaf links (full mesh)")
        for spine in self.SpineSwitchList:
            for leaf in self.LeafSwitchList:
                linkopts = dict(bw=self.bw_spine_leaf)
                self.addLink(spine, leaf, **linkopts)
        
        logger.debug("Creating Leaf to Host links")
        for leaf_idx, leaf in enumerate(self.LeafSwitchList):
            host_start = leaf_idx * self.hosts_per_leaf
            host_end = min(host_start + self.hosts_per_leaf, len(self.HostList))
            
            for host_idx in range(host_start, host_end):
                if host_idx < len(self.HostList):
                    linkopts = dict(bw=self.bw_leaf_host)
                    self.addLink(leaf, self.HostList[host_idx], **linkopts)


class ElasticTreeOptimizer:
    def __init__(self, net, topo, active_ratio=0.5):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self.active_ratio = active_ratio
        self._build_topology_maps()
        self._build_traffic_matrix(active_ratio)
    
    def _build_topology_maps(self):
        """Build mappings between hosts, leaves, and spines"""
        self.host_to_leaf = {}
        self.leaf_to_hosts = defaultdict(list)
        self.leaf_to_spines = defaultdict(list)
        
        for i, host in enumerate(self.topo.HostList):
            leaf_idx = i // self.topo.hosts_per_leaf
            if leaf_idx < len(self.topo.LeafSwitchList):
                leaf = self.topo.LeafSwitchList[leaf_idx]
                self.host_to_leaf[host] = leaf
                self.leaf_to_hosts[leaf].append(host)
        
        for leaf in self.topo.LeafSwitchList:
            self.leaf_to_spines[leaf] = self.topo.SpineSwitchList.copy()
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """
        Build traffic demand matrix between hosts
        active_ratio: fraction of hosts with traffic (0.01 to 1.0)
        """
        self.traffic_matrix = defaultdict(lambda: defaultdict(float))
        
        hosts = [h.name for h in self.net.hosts]
        
        num_active = max(1, int(len(hosts) * active_ratio))
        active_hosts = random.sample(hosts, num_active)
        
        info(f"Active hosts for traffic: {num_active}/{len(hosts)} ({active_ratio*100:.0f}%)\n")
        info(f"Active hosts: {', '.join(sorted(active_hosts))}\n")
        
        for src in active_hosts:
            for dst in active_hosts:
                if src != dst:
                    self.traffic_matrix[src][dst] = random.uniform(0.01, 0.5)
    
    def build_affinity_groups(self):
        """
        Build affinity groups: clusters of hosts that communicate heavily
        These should be co-located on the same leaf switch
        """
        info(f"\n*** Building Communication Affinity Groups ***\n")
        
        host_traffic = defaultdict(float)
        for src in self.traffic_matrix:
            for dst in self.traffic_matrix[src]:
                traffic = self.traffic_matrix[src][dst]
                pair = tuple(sorted([src, dst]))
                host_traffic[pair] += traffic
        
        all_hosts = set()
        for src in self.traffic_matrix:
            all_hosts.add(src)
        for flows in self.traffic_matrix.values():
            for dst in flows:
                all_hosts.add(dst)
        
        sorted_pairs = sorted(host_traffic.items(), key=lambda x: x[1], reverse=True)
        
        affinity_groups = []
        assigned_hosts = set()
        
        for (h1, h2), traffic in sorted_pairs:
            if h1 in assigned_hosts or h2 in assigned_hosts:
                continue
            
            group = {h1, h2}
            assigned_hosts.add(h1)
            assigned_hosts.add(h2)
            
            for host in all_hosts:
                if host in assigned_hosts or len(group) >= self.topo.hosts_per_leaf:
                    break
                
                has_affinity = False
                for member in group:
                    pair = tuple(sorted([host, member]))
                    if pair in host_traffic and host_traffic[pair] > 0.05:
                        has_affinity = True
                        break
                
                if has_affinity:
                    group.add(host)
                    assigned_hosts.add(host)
            
            affinity_groups.append(group)
        
        for host in all_hosts:
            if host not in assigned_hosts:
                affinity_groups.append({host})
        
        info(f"Created {len(affinity_groups)} affinity groups:\n")
        for i, group in enumerate(affinity_groups):
            info(f"  Group {i}: {sorted(group)} (size: {len(group)})\n")
        
        return affinity_groups
    
    def topology_aware_placement(self, affinity_groups):
        """
        Place affinity groups on leaf switches to minimize inter-leaf traffic
        """
        required_switches = set()
        
        info(f"\n*** Topology-Aware Placement ***\n")
        
        sorted_groups = sorted(affinity_groups, key=lambda g: len(g), reverse=True)
        
        leaf_assignment = {}
        leaf_utilization = defaultdict(int)
        
        current_leaf_idx = 0
        
        for group in sorted_groups:
            while current_leaf_idx < len(self.topo.LeafSwitchList):
                leaf = self.topo.LeafSwitchList[current_leaf_idx]
                
                if leaf_utilization[leaf] + len(group) <= self.topo.hosts_per_leaf:
                    leaf_assignment[frozenset(group)] = leaf
                    leaf_utilization[leaf] += len(group)
                    required_switches.add(leaf)
                    
                    group_repr = sorted(group)[:3]
                    if len(group) > 3:
                        group_repr = group_repr + ['...']
                    info(f"  Group {group_repr} -> {leaf} (utilization: {leaf_utilization[leaf]}/{self.topo.hosts_per_leaf})\n")
                    break
                else:
                    current_leaf_idx += 1
            else:
                info(f"WARNING: Not enough leaf capacity for all groups\n")
                leaf = self.topo.LeafSwitchList[0]
                leaf_assignment[frozenset(group)] = leaf
                required_switches.add(leaf)
        
        intra_leaf_traffic = 0
        inter_leaf_traffic = 0
        inter_leaf_flows = defaultdict(float)
        
        for src in self.traffic_matrix:
            src_leaf = None
            for group, leaf in leaf_assignment.items():
                if src in group:
                    src_leaf = leaf
                    break
            if not src_leaf:
                continue
            
            for dst in self.traffic_matrix[src]:
                dst_leaf = None
                for group, leaf in leaf_assignment.items():
                    if dst in group:
                        dst_leaf = leaf
                        break
                if not dst_leaf:
                    continue
                
                traffic = self.traffic_matrix[src][dst]
                
                if src_leaf == dst_leaf:
                    intra_leaf_traffic += traffic
                else:
                    inter_leaf_traffic += traffic
                    flow_key = tuple(sorted([src_leaf, dst_leaf]))
                    inter_leaf_flows[flow_key] += traffic
        
        info(f"\n*** Traffic Analysis ***\n")
        active_leaves = len(required_switches)
        info(f"  Active leaf switches: {active_leaves}/{len(self.topo.LeafSwitchList)}\n")
        info(f"  Intra-leaf traffic: {intra_leaf_traffic:.3f} Gbps (stays within leaf)\n")
        info(f"  Inter-leaf traffic: {inter_leaf_traffic:.3f} Gbps (requires spine)\n")
        
        if inter_leaf_traffic == 0 or active_leaves <= 1:
            num_spines_needed = 0
            info(f"  Spine switches needed: 0 (no inter-leaf traffic)\n")
        else:
            spine_capacity = self.topo.bw_spine_leaf
            num_spines_needed = max(1, int((inter_leaf_traffic / spine_capacity) + 0.5))
            num_spines_needed = min(num_spines_needed, len(self.topo.SpineSwitchList))
            
            info(f"  Spine switches needed: {num_spines_needed} (for {inter_leaf_traffic:.3f} Gbps)\n")
            
            for i in range(num_spines_needed):
                required_switches.add(self.topo.SpineSwitchList[i])
        
        if inter_leaf_flows:
            info(f"\n  Top inter-leaf flows:\n")
            sorted_flows = sorted(inter_leaf_flows.items(), key=lambda x: x[1], reverse=True)
            for (leaf1, leaf2), traffic in sorted_flows[:5]:
                info(f"    {leaf1} <-> {leaf2}: {traffic:.3f} Gbps\n")
        
        return required_switches
    
    def visualize_topology(self):
        """Display the current state of the topology"""
        info("\n" + "="*80 + "\n")
        info("TOPOLOGY STATE (TOPOLOGY-AWARE AFFINITY PLACEMENT)\n")
        info("="*80 + "\n")
        
        info("\nSPINE LAYER:\n")
        for spine in self.topo.SpineSwitchList:
            status = "ON " if spine in self.active_switches else "OFF"
            info(f"  {spine}: {status}\n")
        
        info("\nLEAF LAYER:\n")
        for leaf in self.topo.LeafSwitchList:
            status = "ON " if leaf in self.active_switches else "OFF"
            hosts = self.leaf_to_hosts[leaf]
            info(f"  {leaf}: {status} <- {', '.join(hosts)}\n")
        
        info("="*80 + "\n\n")
    
    def optimize_topology(self):
        """Main optimization routine"""
        info("\n*** ElasticTree Topology-Aware Optimization (Leaf-Spine) ***\n")
        
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        
        affinity_groups = self.build_affinity_groups()
        
        required_switches = self.topology_aware_placement(affinity_groups)
        
        total_switches = len(self.topo.SpineSwitchList) + len(self.topo.LeafSwitchList)
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"\n*** Optimization Results ***\n")
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Power savings:      {power_savings:.1f}%\n")
        
        spine_active = sum(1 for s in self.topo.SpineSwitchList if s in required_switches)
        leaf_active = sum(1 for s in self.topo.LeafSwitchList if s in required_switches)
        
        info(f"\n  Layer Breakdown:\n")
        info(f"    Spine: {spine_active}/{len(self.topo.SpineSwitchList)}\n")
        info(f"    Leaf:  {leaf_active}/{len(self.topo.LeafSwitchList)}\n")
        
        self.visualize_topology()
        
        return self.active_switches


def run_topology():
    setLogLevel('info')
    
    num_spines = 4
    num_leaves = 8
    hosts_per_leaf = 4
    
    topo = LeafSpineTopo(
        num_spines=num_spines,
        num_leaves=num_leaves,
        hosts_per_leaf=hosts_per_leaf
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
    
    info(f"\n*** Leaf-Spine Topology ***\n")
    info(f"  Spine switches:   {topo.num_spines}\n")
    info(f"  Leaf switches:    {topo.num_leaves}\n")
    info(f"  Hosts per leaf:   {topo.hosts_per_leaf}\n")
    info(f"  Total hosts:      {topo.total_hosts}\n")
    
    optimizer = ElasticTreeOptimizer(net, topo, active_ratio=0.5)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()