#!/usr/bin/env python3
"""
ElasticTree Topology-aware (FIXED):
1. Proper left-edge packing that consolidates hosts to leftmost switches
2. Correct pod and core switch selection
3. Visualization to see optimization in action

Usage:
    python3 topology_aware_fixed.py
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import math
from collections import defaultdict

logging.basicConfig(filename='./fattree_elastictree.log', level=logging.DEBUG)
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


class ElasticTreeOptimizer:
    def __init__(self, net, topo):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self._build_topology_maps()
        
    def _build_topology_maps(self):
        self.host_to_edge = {}
        self.edge_to_pod = {}
        self.pod_to_aggs = defaultdict(list)
        self.pod_to_edges = defaultdict(list)
        
        density = int(self.topo.density)
        end = int(self.topo.pod / 2)
        
        for i, host in enumerate(self.topo.HostList):
            edge_idx = i // density
            if edge_idx < len(self.topo.EdgeSwitchList):
                self.host_to_edge[host] = self.topo.EdgeSwitchList[edge_idx]
        
        for i, edge in enumerate(self.topo.EdgeSwitchList):
            pod_idx = i // end
            self.edge_to_pod[edge] = pod_idx
            self.pod_to_edges[pod_idx].append(edge)
            
        for i, agg in enumerate(self.topo.AggSwitchList):
            pod_idx = i // end
            self.pod_to_aggs[pod_idx].append(agg)
    
    def get_active_hosts(self):
        active = set()
        for host in self.net.hosts:
            active.add(host.name)
        return active
    
    def left_edge_packing(self, active_hosts):
        required_switches = set()
        
        if not active_hosts:
            return required_switches
        
        density = int(self.topo.density)
        num_edges_needed = math.ceil(len(active_hosts) / density)
        
        packed_edges = self.topo.EdgeSwitchList[:num_edges_needed]
        for edge in packed_edges:
            required_switches.add(edge)
        
        active_pods = set()
        for edge in packed_edges:
            if edge in self.edge_to_pod:
                active_pods.add(self.edge_to_pod[edge])
        
        for pod in active_pods:
            for agg in self.pod_to_aggs[pod]:
                required_switches.add(agg)
        
        if len(active_pods) == 1:
            num_core_needed = 1
        else:
            num_core_needed = max(2, math.ceil(len(active_pods) / 2))
        
        num_core_needed = min(num_core_needed, len(self.topo.CoreSwitchList))
        
        for i in range(num_core_needed):
            required_switches.add(self.topo.CoreSwitchList[i])
        
        return required_switches
    
    def visualize_topology(self):
        info("\n" + "="*70 + "\n")
        info("CORE LAYER:\n")
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
        
        info("\nEDGE LAYER:\n")
        density = int(self.topo.density)
        for pod in range(self.topo.pod):
            info(f"  Pod {pod}:\n")
            for edge in self.pod_to_edges[pod]:
                status = "ON " if edge in self.active_switches else "OFF"
                edge_idx = self.topo.EdgeSwitchList.index(edge)
                host_start = edge_idx * density
                host_end = min(host_start + density, len(self.topo.HostList))
                hosts = self.topo.HostList[host_start:host_end]
                info(f"    {edge}: {status} <- {', '.join(hosts)}\n")
        
        info("="*70 + "\n\n")
    
    def optimize_topology(self):
        info("\n*** ElasticTree Optimization ***\n")
        
        active_hosts = self.get_active_hosts()        
        required_switches = self.left_edge_packing(active_hosts)
        
        total_switches = (len(self.topo.CoreSwitchList) + 
                         len(self.topo.AggSwitchList) + 
                         len(self.topo.EdgeSwitchList))
        
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
        info(f"\nResults:\n")
        info(f"  Total switches:    {total_switches}\n")
        info(f"  Active switches:   {len(required_switches)}\n")
        
        core_active = sum(1 for s in self.topo.CoreSwitchList if s in required_switches)
        agg_active = sum(1 for s in self.topo.AggSwitchList if s in required_switches)
        edge_active = sum(1 for s in self.topo.EdgeSwitchList if s in required_switches)
        
        info(f"\n  Core:  {core_active}/{len(self.topo.CoreSwitchList)}\n")
        info(f"  Agg:   {agg_active}/{len(self.topo.AggSwitchList)}\n")
        info(f"  Edge:  {edge_active}/{len(self.topo.EdgeSwitchList)}\n")
        
        self.visualize_topology()
        
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
    
    info(f"\n*** FatTree (k={k}):\n")
    info(f"  Core: {topo.iCoreLayerSwitch}, Agg: {topo.iAggLayerSwitch}, ")
    info(f"Edge: {topo.iEdgeLayerSwitch}, Hosts: {topo.iHost}\n")
    
    optimizer = ElasticTreeOptimizer(net, topo)
    optimizer.optimize_topology()
    
    # CLI(net)
    # net.stop()


if __name__ == '__main__':
    run_topology()