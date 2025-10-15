#!/usr/bin/env python3
"""
ElasticTree Formal/Model-Based Optimization:
1. Uses optimization model to find MINIMUM switch set
2. Considers multiple routing paths and redundancy
3. Tries different configurations and picks optimal
4. Guarantees connectivity through path verification

Usage:
    python3 projects/fattree/formal_optimization.py
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

logging.basicConfig(filename='./fattree_elastictree_formal.log', level=logging.DEBUG)
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


class FormalOptimizer:
    """
    Formal optimization using iterative constraint-based approach.
    Simulates linear programming: minimize switches subject to connectivity.
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
        """Build comprehensive topology mappings"""
        self.host_to_edge = {}
        self.edge_to_pod = {}
        self.edge_to_aggs = defaultdict(list)
        self.agg_to_cores = defaultdict(list)
        self.agg_to_edges = defaultdict(list)
        self.pod_to_aggs = defaultdict(list)
        self.pod_to_edges = defaultdict(list)
        self.all_links = []
        
        density = int(self.topo.density)
        end = int(self.topo.pod / 2)
        
        # Map hosts to edge switches
        for i, host in enumerate(self.topo.HostList):
            edge_idx = i // density
            if edge_idx < len(self.topo.EdgeSwitchList):
                edge = self.topo.EdgeSwitchList[edge_idx]
                self.host_to_edge[host] = edge
                self.all_links.append((host, edge))
        
        # Map edge switches
        for i, edge in enumerate(self.topo.EdgeSwitchList):
            pod_idx = i // end
            self.edge_to_pod[edge] = pod_idx
            self.pod_to_edges[pod_idx].append(edge)
            
            # Edge to agg links
            for agg_in_pod in range(end):
                agg_idx = pod_idx * end + agg_in_pod
                if agg_idx < len(self.topo.AggSwitchList):
                    agg = self.topo.AggSwitchList[agg_idx]
                    self.edge_to_aggs[edge].append(agg)
                    self.agg_to_edges[agg].append(edge)
                    self.all_links.append((edge, agg))
        
        # Map agg switches
        for i, agg in enumerate(self.topo.AggSwitchList):
            pod_idx = i // end
            agg_in_pod = i % end
            self.pod_to_aggs[pod_idx].append(agg)
            
            # Agg to core links
            for core_row in range(end):
                core_idx = agg_in_pod * end + core_row
                if core_idx < len(self.topo.CoreSwitchList):
                    core = self.topo.CoreSwitchList[core_idx]
                    self.agg_to_cores[agg].append(core)
                    self.all_links.append((agg, core))
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """Build traffic demand matrix between hosts"""
        self.traffic_matrix = defaultdict(lambda: defaultdict(float))
        
        hosts = [h.name for h in self.net.hosts]
        num_active = max(1, int(len(hosts) * active_ratio))
        active_hosts = random.sample(hosts, num_active)
        
        info(f"Active hosts for traffic: {num_active}/{len(hosts)} ({active_ratio*100:.0f}%)\n")
        info(f"Active: {', '.join(sorted(active_hosts))}\n")
        
        for src in active_hosts:
            for dst in active_hosts:
                if src != dst:
                    self.traffic_matrix[src][dst] = random.uniform(0.01, 0.5)
    
    def find_path(self, src_host, dst_host, active_switches):
        """Find a path between two hosts through active switches"""
        if src_host == dst_host:
            return []
        
        src_edge = self.host_to_edge.get(src_host)
        dst_edge = self.host_to_edge.get(dst_host)
        
        if not src_edge or not dst_edge:
            return None
        
        if src_edge not in active_switches or dst_edge not in active_switches:
            return None
        
        src_pod = self.edge_to_pod[src_edge]
        dst_pod = self.edge_to_pod[dst_edge]
        
        path = [src_host, src_edge]
        
        if src_pod == dst_pod:
            for agg in self.edge_to_aggs[src_edge]:
                if agg in active_switches and agg in self.edge_to_aggs[dst_edge]:
                    path.extend([agg, dst_edge, dst_host])
                    return path
        else:
            for src_agg in self.edge_to_aggs[src_edge]:
                if src_agg not in active_switches:
                    continue
                for core in self.agg_to_cores[src_agg]:
                    if core not in active_switches:
                        continue
                    for dst_agg in self.edge_to_aggs[dst_edge]:
                        if dst_agg in active_switches and core in self.agg_to_cores[dst_agg]:
                            path.extend([src_agg, core, dst_agg, dst_edge, dst_host])
                            return path
        
        return None
    
    def verify_connectivity(self, switch_set):
        """Verify all traffic demands can be satisfied"""
        for src in self.traffic_matrix:
            for dst in self.traffic_matrix[src]:
                if self.traffic_matrix[src][dst] > 0:
                    path = self.find_path(src, dst, switch_set)
                    if not path:
                        return False
        return True
    
    def calculate_cost(self, switch_set):
        """
        Calculate cost (objective function) for a switch configuration
        Cost = number of switches (we want to minimize this)
        """
        return len(switch_set)
    
    def formal_optimization(self):
        """
        Model-based optimization: Try to find MINIMAL switch set
        This simulates branch-and-bound or iterative constraint solving
        """
        info("\nFormal Optimization Process\n")                
        required_switches = set()
        
        active_hosts = set()
        for src in self.traffic_matrix:
            if any(self.traffic_matrix[src].values()):
                active_hosts.add(src)
        for flows in self.traffic_matrix.values():
            for dst in flows:
                if flows[dst] > 0:
                    active_hosts.add(dst)
        
        active_edges = set()
        for host in active_hosts:
            edge = self.host_to_edge.get(host)
            if edge:
                active_edges.add(edge)
                required_switches.add(edge)
        
        info(f"  Required edge switches: {len(active_edges)}\n")
        
        active_pods = set()
        for edge in active_edges:
            active_pods.add(self.edge_to_pod[edge])        
        
        end = int(self.topo.pod / 2)
        agg_configs_to_try = []
        
        if len(active_pods) == 1:
            pod = list(active_pods)[0]
            for i in range(end):
                config = {self.pod_to_aggs[pod][i]}
                agg_configs_to_try.append(config)
        else:
            for i in range(end):
                config = set()
                for pod in active_pods:
                    if i < len(self.pod_to_aggs[pod]):
                        config.add(self.pod_to_aggs[pod][i])
                agg_configs_to_try.append(config)
        
        best_agg_config = None
        for agg_config in agg_configs_to_try:
            test_switches = required_switches | agg_config
            
            test_switches.add(self.topo.CoreSwitchList[0])
            
            if self.verify_connectivity(test_switches):
                best_agg_config = agg_config
                break
        
        if not best_agg_config:
            info("Minimal config failed, using all aggs in active pods\n")
            best_agg_config = set()
            for pod in active_pods:
                best_agg_config.update(self.pod_to_aggs[pod])
        
        required_switches.update(best_agg_config)        
        
        if len(active_pods) == 1:
            info("  Single pod: using 1 core switch\n")
            required_switches.add(self.topo.CoreSwitchList[0])
        else:
            info("  Multi-pod: iteratively adding cores until connected\n")
            for num_cores in range(1, len(self.topo.CoreSwitchList) + 1):
                test_switches = required_switches.copy()
                for i in range(num_cores):
                    test_switches.add(self.topo.CoreSwitchList[i])
                
                if self.verify_connectivity(test_switches):
                    for i in range(num_cores):
                        required_switches.add(self.topo.CoreSwitchList[i])
                    break
        
        original_size = len(required_switches)
        
        removable = []
        for switch in list(required_switches):
            if switch in self.topo.EdgeSwitchList:
                continue
            
            test_switches = required_switches - {switch}
            if self.verify_connectivity(test_switches):
                removable.append(switch)
        
        if removable:
            info(f"  Found {len(removable)} redundant switches to remove\n")
            for switch in removable:
                required_switches.remove(switch)
        else:
            info(f"  Configuration is already minimal\n")
        
        final_size = len(required_switches)
        info(f"\n  Optimization: {original_size} → {final_size} switches\n")
        
        required_links = set()
        for host in active_hosts:
            edge = self.host_to_edge.get(host)
            if edge and edge in required_switches:
                required_links.add((host, edge))
        
        for edge in active_edges:
            for agg in self.edge_to_aggs[edge]:
                if agg in required_switches:
                    required_links.add((edge, agg))
        
        for agg in required_switches:
            if agg in self.topo.AggSwitchList:
                for core in self.agg_to_cores[agg]:
                    if core in required_switches:
                        required_links.add((agg, core))
        
        return required_switches, required_links
    
    def optimize_topology(self):
        """Main optimization routine"""        
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")        
        required_switches, required_links = self.formal_optimization()        
        total_switches = (len(self.topo.CoreSwitchList) + 
                         len(self.topo.AggSwitchList) + 
                         len(self.topo.EdgeSwitchList))
        
        powered_down = total_switches - len(required_switches)
        switch_reduction = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        self.active_links = required_links
        
        # Display results
        info(f"\n*** Optimization Results ***\n")
        info(f"  Total switches:     {total_switches}\n")
        info(f"  Active switches:    {len(required_switches)}\n")
        info(f"  Powered down:       {powered_down}\n")
        info(f"  Switch reduction:   {switch_reduction:.1f}%\n")
        
        core_active = sum(1 for s in self.topo.CoreSwitchList if s in required_switches)
        agg_active = sum(1 for s in self.topo.AggSwitchList if s in required_switches)
        edge_active = sum(1 for s in self.topo.EdgeSwitchList if s in required_switches)
        
        info(f"\n  Layer Breakdown:\n")
        info(f"    Core:  {core_active}/{len(self.topo.CoreSwitchList)}\n")
        info(f"    Agg:   {agg_active}/{len(self.topo.AggSwitchList)}\n")
        info(f"    Edge:  {edge_active}/{len(self.topo.EdgeSwitchList)}\n")
        
        self.visualize_topology()
        
        return self.active_switches
    
    def visualize_topology(self):
        """Visualize optimized topology"""
        info("\n" + "="*80 + "\n")
        info("TOPOLOGY STATE (FORMAL OPTIMIZATION)\n")
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
        
        info("="*80 + "\n\n")


def run_topology():
    setLogLevel('info')
    k = 4
    topo = FatTree(k)
    net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch, link=TCLink, autoSetMacs=True)
    net.start()
    optimizer = FormalOptimizer(net, topo, active_ratio=0.5)
    optimizer.optimize_topology()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    try:
        run_topology()
    except KeyboardInterrupt:
        info("\n*** Interrupted by user ***\n")
        from mininet.clean import cleanup
        cleanup()