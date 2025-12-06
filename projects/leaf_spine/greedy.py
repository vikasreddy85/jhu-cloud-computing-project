#!/usr/bin/env python3
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

class LeafSpineTopo(Topo):
    def __init__(self, num_spines=4, num_leaves=8, hosts_per_leaf=4, 
                 bw_spine_leaf=10.0, bw_host_leaf=1.0):
        self.num_spines = num_spines
        self.num_leaves = num_leaves
        self.hosts_per_leaf = hosts_per_leaf
        self.total_hosts = num_leaves * hosts_per_leaf
        
        self.bw_spine_leaf = bw_spine_leaf
        self.bw_host_leaf = bw_host_leaf
        
        self.SpineList = []
        self.LeafList = []
        self.HostList = []
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createLinks()
    
    def createTopo(self):
        """Create all switches and hosts"""
        for i in range(1, self.num_spines + 1):
            spine_name = f's{i:03d}'
            self.SpineList.append(self.addSwitch(spine_name))
        
        for i in range(1, self.num_leaves + 1):
            leaf_name = f'l{i:03d}'
            self.LeafList.append(self.addSwitch(leaf_name))
        
        for i in range(1, self.total_hosts + 1):
            if i < 10:
                host_name = f'h00{i}'
            elif i < 100:
                host_name = f'h0{i}'
            else:
                host_name = f'h{i}'
            self.HostList.append(self.addHost(host_name))
    
    def createLinks(self):
        """Create links between layers"""
        logger.debug("Creating Spine to Leaf links (full mesh)")
        
        for leaf in self.LeafList:
            for spine in self.SpineList:
                linkopts = dict(bw=self.bw_spine_leaf)
                self.addLink(spine, leaf, **linkopts)
        
        logger.debug("Creating Leaf to Host links")
        
        for leaf_idx, leaf in enumerate(self.LeafList):
            host_start = leaf_idx * self.hosts_per_leaf
            host_end = min(host_start + self.hosts_per_leaf, len(self.HostList))
            
            for host_idx in range(host_start, host_end):
                if host_idx < len(self.HostList):
                    linkopts = dict(bw=self.bw_host_leaf)
                    self.addLink(leaf, self.HostList[host_idx], **linkopts)


class GreedyBinPackingOptimizer:
    """Optimizer for leaf-spine topology using greedy bin-packing"""
    
    def __init__(self, net, topo):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self._build_topology_maps()
        
    def _build_topology_maps(self):
        """Build mappings between topology elements"""
        self.host_to_leaf = {}
        self.leaf_to_spines = defaultdict(list)
        
        for i, host in enumerate(self.topo.HostList):
            leaf_idx = i // self.topo.hosts_per_leaf
            if leaf_idx < len(self.topo.LeafList):
                self.host_to_leaf[host] = self.topo.LeafList[leaf_idx]
        
        for leaf in self.topo.LeafList:
            self.leaf_to_spines[leaf] = self.topo.SpineList.copy()
    
    def get_host_traffic(self):
        """Simulate host traffic (in real implementation, would query controller)"""
        traffic = {}
        for host in self.net.hosts:
            traffic[host.name] = random.uniform(0.01, 0.5)
        return traffic
    
    def greedy_bin_packing(self, host_traffic):
        required_switches = set()
        
        if not host_traffic:
            return required_switches, {}
        
        sorted_hosts = sorted(host_traffic.items(), key=lambda x: x[1], reverse=True)
        
        leaf_utilization = defaultdict(float)
        leaf_capacity = self.topo.bw_host_leaf * self.topo.hosts_per_leaf
        host_to_assigned_leaf = {}

        for host_name, traffic in sorted_hosts:
            assigned = False
            
            for leaf in self.topo.LeafList:
                if leaf in required_switches:
                    if leaf_utilization[leaf] + traffic <= leaf_capacity:
                        leaf_utilization[leaf] += traffic
                        host_to_assigned_leaf[host_name] = leaf
                        assigned = True
                        info(f"  {host_name} ({traffic:.3f} Gbps) -> {leaf} (util: {leaf_utilization[leaf]:.3f}/{leaf_capacity:.2f})\n")
                        break
            
            if not assigned:
                for leaf in self.topo.LeafList:
                    if leaf not in required_switches:
                        leaf_utilization[leaf] = traffic
                        required_switches.add(leaf)
                        host_to_assigned_leaf[host_name] = leaf
                        info(f"  {host_name} ({traffic:.3f} Gbps) -> {leaf} [NEW] (util: {leaf_utilization[leaf]:.3f}/{leaf_capacity:.2f})\n")
                        break
        
        active_leaves = [l for l in required_switches if l in self.topo.LeafList]
        
        if len(active_leaves) == 0:
            num_spines_needed = 0
        elif len(active_leaves) == 1:
            num_spines_needed = 0
        else:
            num_spines_needed = min(2, len(self.topo.SpineList))
        
        for i in range(num_spines_needed):
            if i < len(self.topo.SpineList):
                required_switches.add(self.topo.SpineList[i])
        
        return required_switches, leaf_utilization
    
    def visualize_topology(self, leaf_utilization):
        """Visualize the topology with utilization information"""
        info("\nspine:\n")
        for i, spine in enumerate(self.topo.SpineList, 1):
            status = "ON " if spine in self.active_switches else "OFF"
            info(f"  Spine {i} ({spine}): {status}\n")
        
        info("\nleaf:\n")
        leaf_capacity = self.topo.bw_host_leaf * self.topo.hosts_per_leaf
        
        for i, leaf in enumerate(self.topo.LeafList, 1):
            status = "ON " if leaf in self.active_switches else "OFF"
            
            host_start = (i - 1) * self.topo.hosts_per_leaf
            host_end = min(host_start + self.topo.hosts_per_leaf, len(self.topo.HostList))
            hosts = self.topo.HostList[host_start:host_end]
            
            if leaf in leaf_utilization and leaf in self.active_switches:
                util = leaf_utilization[leaf]
                util_pct = (util / leaf_capacity) * 100
                info(f"  Leaf {i} ({leaf}): {status} [Util: {util:.3f}/{leaf_capacity:.2f} Gbps = {util_pct:.1f}%]\n")
            else:
                info(f"  Leaf {i} ({leaf}): {status}\n")
            
            info(f"         Hosts: {', '.join(hosts)}\n")
        
    
    def optimize_topology(self):
        """Main optimization routine"""        
        host_traffic = self.get_host_traffic()
        total_traffic = sum(host_traffic.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        info(f"Active hosts: {len(host_traffic)}\n")
        
        required_switches, leaf_utilization = self.greedy_bin_packing(host_traffic)
        
        total_switches = len(self.topo.SpineList) + len(self.topo.LeafList)
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Power savings:      {power_savings:.1f}%\n")
        
        spine_active = sum(1 for s in self.topo.SpineList if s in required_switches)
        leaf_active = sum(1 for s in self.topo.LeafList if s in required_switches)
        
        info(f"\n  Layer Breakdown:\n")
        info(f"    Spine: {spine_active}/{len(self.topo.SpineList)}\n")
        info(f"    Leaf:  {leaf_active}/{len(self.topo.LeafList)}\n")
        
        if leaf_active > 0:
            active_leaf_utils = [leaf_utilization[l] for l in self.topo.LeafList 
                                if l in required_switches]
            avg_util = sum(active_leaf_utils) / len(active_leaf_utils) if active_leaf_utils else 0
            leaf_capacity = self.topo.bw_host_leaf * self.topo.hosts_per_leaf
            avg_util_pct = (avg_util / leaf_capacity) * 100
            info(f"\n  Average leaf utilization: {avg_util:.3f} Gbps ({avg_util_pct:.1f}%)\n")
        
        self.visualize_topology(leaf_utilization)
        
        return self.active_switches


def run_topology():
    """Run the leaf-spine topology with greedy optimization"""
    setLogLevel('info')
    
    num_spines = 4
    num_leaves = 8
    hosts_per_leaf = 4
    
    topo = LeafSpineTopo(
        num_spines=num_spines,
        num_leaves=num_leaves,
        hosts_per_leaf=hosts_per_leaf,
        bw_spine_leaf=10.0,
        bw_host_leaf=1.0
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
    
    info(f"  Spine switches:   {num_spines}\n")
    info(f"  Leaf switches:    {num_leaves}\n")
    info(f"  Hosts per leaf:   {hosts_per_leaf}\n")
    info(f"  Total hosts:      {topo.total_hosts}\n")
    info(f"  Spine-Leaf BW:    {topo.bw_spine_leaf} Gbps\n")
    info(f"  Host-Leaf BW:     {topo.bw_host_leaf} Gbps\n")
    
    optimizer = GreedyBinPackingOptimizer(net, topo)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()