#!/usr/bin/env python3
"""
ElasticTree Greedy Bin-Packing:
1. Sorts hosts by traffic/utilization
2. Greedily packs hosts into switches to maximize consolidation
3. Only activates switches needed for current traffic patterns
4. Considers link capacity constraints

Usage:
    python3 projects/fattree/greedy_binpacking.py
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

logging.basicConfig(filename='./fattree_elastictree_greedy.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FatTree(Topo):
    def __init__(self, k=4):
        self.pod = k
        self.iCoreLayerSwitch = int((k/2)**2)
        self.iAggLayerSwitch = int(k*k/2)
        self.iEdgeLayerSwitch = int(k*k/2)
        self.density = int(k/2)
        self.iHost = int(self.iEdgeLayerSwitch * self.density)
        
        self.bw_c2a = 1.0
        self.bw_a2e = 1.0
        self.bw_h2a = 1.0
        
        self.CoreSwitchList = []
        self.AggSwitchList = []
        self.EdgeSwitchList = []
        self.HostList = []
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createLink(bw_c2a=self.bw_c2a, 
                       bw_a2e=self.bw_a2e, 
                       bw_h2a=self.bw_h2a)
    
    def createTopo(self):
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)
    
    def _addSwitch(self, number, level, switch_list):
        for x in range(1, int(number)+1):
            PREFIX = str(level) + "00"
            if x >= 10:
                PREFIX = str(level) + "0"
            switch_list.append(self.addSwitch('s' + PREFIX + str(x)))
    
    def createCoreLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 1, self.CoreSwitchList)
    
    def createAggLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 2, self.AggSwitchList)
    
    def createEdgeLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 3, self.EdgeSwitchList)
    
    def createHost(self, NUMBER):
        for x in range(1, int(NUMBER)+1):
            PREFIX = "h00"
            if x >= 10:
                PREFIX = "h0"
            elif x >= 100:
                PREFIX = "h"
            self.HostList.append(self.addHost(PREFIX + str(x)))
    
    def createLink(self, bw_c2a=1.0, bw_a2e=1.0, bw_h2a=1.0):
        end = int(self.pod/2)
        
        logger.debug("Creating Core to Agg links")
        for pod in range(self.pod):
            for agg_in_pod in range(end):
                agg_idx = pod * end + agg_in_pod
                for core_row in range(end):
                    core_idx = agg_in_pod * end + core_row
                    if core_idx < len(self.CoreSwitchList) and agg_idx < len(self.AggSwitchList):
                        linkopts = dict(bw=bw_c2a)
                        self.addLink(
                            self.CoreSwitchList[core_idx],
                            self.AggSwitchList[agg_idx],
                            **linkopts)
        
        logger.debug("Creating Agg to Edge links")
        for pod in range(self.pod):
            for agg_in_pod in range(end):
                agg_idx = pod * end + agg_in_pod
                for edge_in_pod in range(end):
                    edge_idx = pod * end + edge_in_pod
                    if agg_idx < len(self.AggSwitchList) and edge_idx < len(self.EdgeSwitchList):
                        linkopts = dict(bw=bw_a2e)
                        self.addLink(
                            self.AggSwitchList[agg_idx], 
                            self.EdgeSwitchList[edge_idx],
                            **linkopts)
        
        logger.debug("Creating Edge to Host links")
        for edge_idx in range(len(self.EdgeSwitchList)):
            for host_in_edge in range(int(self.density)):
                host_idx = edge_idx * int(self.density) + host_in_edge
                if host_idx < len(self.HostList):
                    linkopts = dict(bw=bw_h2a)
                    self.addLink(
                        self.EdgeSwitchList[edge_idx],
                        self.HostList[host_idx],
                        **linkopts)


class GreedyBinPackingOptimizer:
    def __init__(self, net, topo):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self._build_topology_maps()
        
    def _build_topology_maps(self):
        """Build mappings between topology elements"""
        self.host_to_edge = {}
        self.edge_to_pod = {}
        self.edge_to_aggs = defaultdict(list)
        self.agg_to_cores = defaultdict(list)
        self.pod_to_aggs = defaultdict(list)
        self.pod_to_edges = defaultdict(list)
        
        density = int(self.topo.density)
        end = int(self.topo.pod / 2)
        
        # Map hosts to edge switches
        for i, host in enumerate(self.topo.HostList):
            edge_idx = i // density
            if edge_idx < len(self.topo.EdgeSwitchList):
                self.host_to_edge[host] = self.topo.EdgeSwitchList[edge_idx]
        
        # Map edge switches to pods and aggregation switches
        for i, edge in enumerate(self.topo.EdgeSwitchList):
            pod_idx = i // end
            self.edge_to_pod[edge] = pod_idx
            self.pod_to_edges[pod_idx].append(edge)
            
            # Each edge connects to all agg switches in its pod
            for agg_in_pod in range(end):
                agg_idx = pod_idx * end + agg_in_pod
                if agg_idx < len(self.topo.AggSwitchList):
                    self.edge_to_aggs[edge].append(self.topo.AggSwitchList[agg_idx])
        
        # Map agg switches to pods and core switches
        for i, agg in enumerate(self.topo.AggSwitchList):
            pod_idx = i // end
            agg_in_pod = i % end
            self.pod_to_aggs[pod_idx].append(agg)
            
            # Each agg connects to specific core switches
            for core_row in range(end):
                core_idx = agg_in_pod * end + core_row
                if core_idx < len(self.topo.CoreSwitchList):
                    self.agg_to_cores[agg].append(self.topo.CoreSwitchList[core_idx])
    
    def get_host_traffic(self):
        """Simulate or measure host traffic (in real implementation, would query controller)"""
        traffic = {}
        for host in self.net.hosts:
            # Simulate traffic: random value between 0.01 and 0.5 Gbps
            traffic[host.name] = random.uniform(0.01, 0.5)
        return traffic
    
    def greedy_bin_packing(self, host_traffic):
        """
        Greedy bin-packing algorithm:
        1. Sort hosts by traffic (descending)
        2. Pack hosts into edge switches, respecting capacity
        3. Activate only necessary upper-layer switches
        """
        required_switches = set()
        
        if not host_traffic:
            return required_switches
        
        # Sort hosts by traffic (largest first for better bin packing)
        sorted_hosts = sorted(host_traffic.items(), key=lambda x: x[1], reverse=True)
        
        edge_utilization = defaultdict(float)
        edge_capacity = self.topo.bw_h2a * self.topo.density  # Total capacity per edge
        host_to_assigned_edge = {}
        
        info(f"\n*** Greedy Bin-Packing Process ***\n")
        info(f"Edge capacity: {edge_capacity:.2f} Gbps\n\n")
        
        # Greedy assignment: try to fill each edge switch before moving to next
        for host_name, traffic in sorted_hosts:
            assigned = False
            
            # Try to fit into an already-active edge switch
            for edge in self.topo.EdgeSwitchList:
                if edge in required_switches:
                    if edge_utilization[edge] + traffic <= edge_capacity:
                        edge_utilization[edge] += traffic
                        host_to_assigned_edge[host_name] = edge
                        assigned = True
                        info(f"  {host_name} ({traffic:.3f} Gbps) -> {edge} (util: {edge_utilization[edge]:.3f}/{edge_capacity:.2f})\n")
                        break
            
            # If not assigned, activate a new edge switch
            if not assigned:
                for edge in self.topo.EdgeSwitchList:
                    if edge not in required_switches:
                        edge_utilization[edge] = traffic
                        required_switches.add(edge)
                        host_to_assigned_edge[host_name] = edge
                        info(f"  {host_name} ({traffic:.3f} Gbps) -> {edge} [NEW] (util: {edge_utilization[edge]:.3f}/{edge_capacity:.2f})\n")
                        break
        
        # Activate necessary aggregation switches
        active_pods = set()
        active_edges = [e for e in required_switches if e in self.topo.EdgeSwitchList]
        
        for edge in active_edges:
            pod = self.edge_to_pod[edge]
            active_pods.add(pod)
            for agg in self.edge_to_aggs[edge]:
                required_switches.add(agg)
        
        if len(active_pods) <= 1:
            num_core_needed = 1
        else:
            num_core_needed = min(len(active_pods), len(self.topo.CoreSwitchList))
        
        for i in range(num_core_needed):
            required_switches.add(self.topo.CoreSwitchList[i])
        
        info(f"\n*** Active Pods: {sorted(active_pods)} ***\n")
        
        return required_switches, edge_utilization
    
    def visualize_topology(self, edge_utilization):
        """Visualize the topology with utilization information"""
        info("\n" + "="*80 + "\n")
        info("TOPOLOGY STATE (GREEDY BIN-PACKING)\n")
        info("="*80 + "\n")
        
        info("\nCORE LAYER:\n")
        for switch in self.topo.CoreSwitchList:
            status = "ON " if switch in self.active_switches else "OFF"
            info(f"  {switch}: {status}\n")
        
        info("\nAGGREGATION LAYER:\n")
        end = int(self.topo.pod / 2)
        for pod in range(self.topo.pod):
            info(f"  Pod {pod}: ")
            statuses = []
            for agg in self.pod_to_aggs[pod]:
                status = "ON" if agg in self.active_switches else "OFF"
                statuses.append(f"{agg}:{status}")
            info(", ".join(statuses) + "\n")
        
        info("\nEDGE LAYER (with utilization):\n")
        density = int(self.topo.density)
        edge_capacity = self.topo.bw_h2a * density
        
        for pod in range(self.topo.pod):
            info(f"  Pod {pod}:\n")
            for edge in self.pod_to_edges[pod]:
                status = "ON " if edge in self.active_switches else "OFF"
                edge_idx = self.topo.EdgeSwitchList.index(edge)
                host_start = edge_idx * density
                host_end = min(host_start + density, len(self.topo.HostList))
                hosts = self.topo.HostList[host_start:host_end]
                
                if edge in edge_utilization and edge in self.active_switches:
                    util = edge_utilization[edge]
                    util_pct = (util / edge_capacity) * 100
                    info(f"    {edge}: {status} [Util: {util:.3f}/{edge_capacity:.2f} Gbps = {util_pct:.1f}%]\n")
                else:
                    info(f"    {edge}: {status}\n")
                    
                info(f"           Hosts: {', '.join(hosts)}\n")
        
        info("="*80 + "\n\n")
    
    def optimize_topology(self):
        """Main optimization routine"""
        info("\n*** ElasticTree Greedy Bin-Packing Optimization ***\n")
        
        # Get host traffic information
        host_traffic = self.get_host_traffic()
        total_traffic = sum(host_traffic.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        info(f"Active hosts: {len(host_traffic)}\n")
        
        # Run greedy bin-packing
        required_switches, edge_utilization = self.greedy_bin_packing(host_traffic)
        
        # Calculate statistics
        total_switches = (len(self.topo.CoreSwitchList) + 
                         len(self.topo.AggSwitchList) + 
                         len(self.topo.EdgeSwitchList))
        
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"\n*** Optimization Results ***\n")
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Power savings:      {power_savings:.1f}%\n")
        
        core_active = sum(1 for s in self.topo.CoreSwitchList if s in required_switches)
        agg_active = sum(1 for s in self.topo.AggSwitchList if s in required_switches)
        edge_active = sum(1 for s in self.topo.EdgeSwitchList if s in required_switches)
        
        info(f"\n  Layer Breakdown:\n")
        info(f"    Core:  {core_active}/{len(self.topo.CoreSwitchList)}\n")
        info(f"    Agg:   {agg_active}/{len(self.topo.AggSwitchList)}\n")
        info(f"    Edge:  {edge_active}/{len(self.topo.EdgeSwitchList)}\n")
        
        if edge_active > 0:
            active_edge_utils = [edge_utilization[e] for e in self.topo.EdgeSwitchList 
                                if e in required_switches]
            avg_util = sum(active_edge_utils) / len(active_edge_utils) if active_edge_utils else 0
            edge_capacity = self.topo.bw_h2a * self.topo.density
            avg_util_pct = (avg_util / edge_capacity) * 100
            info(f"\n  Average edge utilization: {avg_util:.3f} Gbps ({avg_util_pct:.1f}%)\n")
        
        self.visualize_topology(edge_utilization)
        
        return self.active_switches


def run_topology():
    setLogLevel('info')
    k = 4
    topo = FatTree(k=k)
    
    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )
    
    net.start()
    
    info(f"\n*** FatTree Topology (k={k}) ***\n")
    info(f"  Core switches:    {topo.iCoreLayerSwitch}\n")
    info(f"  Agg switches:     {topo.iAggLayerSwitch}\n")
    info(f"  Edge switches:    {topo.iEdgeLayerSwitch}\n")
    info(f"  Hosts:            {topo.iHost}\n")
    info(f"  Hosts per edge:   {topo.density}\n")
    
    optimizer = GreedyBinPackingOptimizer(net, topo)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()