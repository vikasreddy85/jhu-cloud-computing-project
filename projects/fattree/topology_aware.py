from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import logging
import math
import random
from collections import defaultdict

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
    def __init__(self, net, topo, active_ratio=0.5):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self.active_ratio = active_ratio
        self._build_topology_maps()
        self._build_traffic_matrix(active_ratio)
        
    def _build_topology_maps(self):
        self.host_to_edge = {}
        self.edge_to_pod = {}
        self.pod_to_aggs = defaultdict(list)
        self.pod_to_edges = defaultdict(list)
        self.edge_to_aggs = defaultdict(list)
        self.agg_to_cores = defaultdict(list)
        
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
            
            for agg_in_pod in range(end):
                agg_idx = pod_idx * end + agg_in_pod
                if agg_idx < len(self.topo.AggSwitchList):
                    self.edge_to_aggs[edge].append(self.topo.AggSwitchList[agg_idx])
        
        for i, agg in enumerate(self.topo.AggSwitchList):
            pod_idx = i // end
            agg_in_pod = i % end
            self.pod_to_aggs[pod_idx].append(agg)
            
            for core_row in range(end):
                core_idx = agg_in_pod * end + core_row
                if core_idx < len(self.topo.CoreSwitchList):
                    self.agg_to_cores[agg].append(self.topo.CoreSwitchList[core_idx])
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """
        Build traffic demand matrix between hosts
        active_ratio: fraction of hosts with traffic (0.01 to 0.5)
        """
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
    
    def build_affinity_groups(self):
        """
        Build affinity groups: clusters of hosts that communicate heavily
        This is the key topology-aware insight
        """
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
        
        density = int(self.topo.density)
        
        for (h1, h2), traffic in sorted_pairs:
            if h1 in assigned_hosts or h2 in assigned_hosts:
                continue
            
            group = {h1, h2}
            assigned_hosts.add(h1)
            assigned_hosts.add(h2)
            
            for host in all_hosts:
                if host in assigned_hosts or len(group) >= density:
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
        Place affinity groups intelligently across the topology
        """
        required_switches = set()        
        density = int(self.topo.density)
        end = int(self.topo.pod / 2)
        
        sorted_groups = sorted(affinity_groups, key=lambda g: len(g), reverse=True)
        
        edge_assignment = {}
        pod_utilization = defaultdict(int)
        
        current_pod = 0
        edge_in_pod = 0
        
        for group in sorted_groups:
            if current_pod >= self.topo.pod:
                info(f"Not enough pods, wrapping around\n")
                current_pod = 0
                edge_in_pod = 0
            
            edge_idx = current_pod * end + edge_in_pod
            if edge_idx >= len(self.topo.EdgeSwitchList):
                edge_idx = len(self.topo.EdgeSwitchList) - 1
            
            edge = self.topo.EdgeSwitchList[edge_idx]
            edge_assignment[frozenset(group)] = edge
            required_switches.add(edge)
            pod_utilization[current_pod] += 1
            
            info(f"Group {sorted(group)[:3]}{'...' if len(group) > 3 else ''} -> {edge} (Pod {current_pod})\n")
            
            edge_in_pod += 1
            if edge_in_pod >= end:
                edge_in_pod = 0
                current_pod += 1
        
        active_pods = set()
        for edge in required_switches:
            if edge in self.topo.EdgeSwitchList:
                pod = self.edge_to_pod[edge]
                active_pods.add(pod)
        
        for pod in active_pods:
            for agg in self.pod_to_aggs[pod]:
                required_switches.add(agg)
        
        inter_pod_traffic = 0
        intra_pod_traffic = 0
        
        for src in self.traffic_matrix:
            src_edge = None
            for group, edge in edge_assignment.items():
                if src in group:
                    src_edge = edge
                    break
            if not src_edge:
                continue
            src_pod = self.edge_to_pod[src_edge]
            
            for dst in self.traffic_matrix[src]:
                dst_edge = None
                for group, edge in edge_assignment.items():
                    if dst in group:
                        dst_edge = edge
                        break
                if not dst_edge:
                    continue
                dst_pod = self.edge_to_pod[dst_edge]
                
                traffic = self.traffic_matrix[src][dst]
                if src_pod == dst_pod:
                    intra_pod_traffic += traffic
                else:
                    inter_pod_traffic += traffic
        
        info(f"  Active pods: {sorted(active_pods)}\n")
        info(f"  Intra-pod traffic: {intra_pod_traffic:.3f} Gbps\n")
        info(f"  Inter-pod traffic: {inter_pod_traffic:.3f} Gbps\n")
        
        if len(active_pods) <= 1:
            num_core_needed = 1
            info(f"Core switches needed: {num_core_needed} (single pod)\n")
        else:
            total_capacity = self.topo.bw_c2a
            num_core_needed = max(1, int((inter_pod_traffic / total_capacity) + 0.5))
            num_core_needed = min(num_core_needed, len(self.topo.CoreSwitchList))
            info(f"Core switches needed: {num_core_needed} (for {inter_pod_traffic:.3f} Gbps)\n")
        
        for i in range(num_core_needed):
            required_switches.add(self.topo.CoreSwitchList[i])
        
        return required_switches
    
    def visualize_topology(self):
        info("\Core Layer:\n")
        for switch in self.topo.CoreSwitchList:
            status = "ON " if switch in self.active_switches else "OFF"
            info(f"  {switch}: {status}\n")
        
        info("\Aggregation Layer:\n")
        end = int(self.topo.pod / 2)
        for pod in range(self.topo.pod):
            info(f"  Pod {pod}: ")
            statuses = []
            for agg in self.pod_to_aggs[pod]:
                status = "ON" if agg in self.active_switches else "OFF"
                statuses.append(f"{agg}:{status}")
            info(", ".join(statuses) + "\n")
        
        info("\Edge Layer:\n")
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
            
    def optimize_topology(self):        
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"Total network traffic: {total_traffic:.3f} Gbps\n")
        
        affinity_groups = self.build_affinity_groups()
        
        required_switches = self.topology_aware_placement(affinity_groups)
        
        total_switches = (len(self.topo.CoreSwitchList) + 
                         len(self.topo.AggSwitchList) + 
                         len(self.topo.EdgeSwitchList))
        
        powered_down = total_switches - len(required_switches)
        power_savings = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        
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
    
    info(f"  Core switches:    {topo.iCoreLayerSwitch}\n")
    info(f"  Agg switches:     {topo.iAggLayerSwitch}\n")
    info(f"  Edge switches:    {topo.iEdgeLayerSwitch}\n")
    info(f"  Hosts:            {topo.iHost}\n")
    info(f"  Hosts per edge:   {topo.density}\n")
    
    optimizer = ElasticTreeOptimizer(net, topo, active_ratio=0.5)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    run_topology()