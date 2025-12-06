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
    def __init__(self, num_spines=4, num_leaves=4, hosts_per_leaf=4):
        self.num_spines = num_spines
        self.num_leaves = num_leaves
        self.hosts_per_leaf = hosts_per_leaf
        self.total_hosts = num_leaves * hosts_per_leaf
        
        self.bw_spine_leaf = 10.0
        self.bw_host_leaf = 1.0
        
        self.SpineSwitchList = []
        self.LeafSwitchList = []
        self.HostList = []
        
        Topo.__init__(self)
        
        self.createTopo()
        self.createLink()
    
    def createTopo(self):
        """Create all switches and hosts"""
        for i in range(1, self.num_spines + 1):
            switch_name = f's10{i:02d}'
            self.SpineSwitchList.append(self.addSwitch(switch_name))
        
        for i in range(1, self.num_leaves + 1):
            switch_name = f's20{i:02d}'
            self.LeafSwitchList.append(self.addSwitch(switch_name))
        
        for i in range(1, self.total_hosts + 1):
            if i < 10:
                host_name = f'h00{i}'
            elif i < 100:
                host_name = f'h0{i}'
            else:
                host_name = f'h{i}'
            self.HostList.append(self.addHost(host_name))
    
    def createLink(self):
        """Create links: full mesh spine-leaf, and leaf-host connections"""
        logger.debug("Creating Spine to Leaf links (full mesh)")
        
        for leaf in self.LeafSwitchList:
            for spine in self.SpineSwitchList:
                linkopts = dict(bw=self.bw_spine_leaf)
                self.addLink(leaf, spine, **linkopts)
        
        logger.debug("Creating Leaf to Host links")
        
        for leaf_idx, leaf in enumerate(self.LeafSwitchList):
            start_host = leaf_idx * self.hosts_per_leaf
            end_host = min(start_host + self.hosts_per_leaf, len(self.HostList))
            
            for host_idx in range(start_host, end_host):
                host = self.HostList[host_idx]
                linkopts = dict(bw=self.bw_host_leaf)
                self.addLink(leaf, host, **linkopts)


class LeafSpineFormalOptimizer:
    def __init__(self, net, topo, active_ratio=0.5):
        self.net = net
        self.topo = topo
        self.active_switches = set()
        self.active_links = set()
        self.active_ratio = active_ratio
        self._build_topology_maps()
        self._build_traffic_matrix(active_ratio)
    
    def _build_topology_maps(self):
        """Build comprehensive topology mappings for leaf-spine"""
        self.host_to_leaf = {}
        self.leaf_to_hosts = defaultdict(list)
        self.leaf_to_spines = defaultdict(list)
        self.spine_to_leaves = defaultdict(list)
        self.all_links = []
        
        for leaf_idx, leaf in enumerate(self.topo.LeafSwitchList):
            start_host = leaf_idx * self.topo.hosts_per_leaf
            end_host = min(start_host + self.topo.hosts_per_leaf, len(self.topo.HostList))
            
            for host_idx in range(start_host, end_host):
                host = self.topo.HostList[host_idx]
                self.host_to_leaf[host] = leaf
                self.leaf_to_hosts[leaf].append(host)
                self.all_links.append((host, leaf))
        
        for leaf in self.topo.LeafSwitchList:
            for spine in self.topo.SpineSwitchList:
                self.leaf_to_spines[leaf].append(spine)
                self.spine_to_leaves[spine].append(leaf)
                self.all_links.append((leaf, spine))
    
    def _build_traffic_matrix(self, active_ratio=0.5):
        """Build traffic demand matrix between hosts"""
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
    
    def find_path(self, src_host, dst_host, active_switches):
        """
        Find a path between two hosts through active switches.
        """
        if src_host == dst_host:
            return []
        
        src_leaf = self.host_to_leaf.get(src_host)
        dst_leaf = self.host_to_leaf.get(dst_host)
        
        if not src_leaf or not dst_leaf:
            return None
        
        if src_leaf not in active_switches or dst_leaf not in active_switches:
            return None
        
        path = [src_host, src_leaf]
        
        if src_leaf == dst_leaf:
            path.extend([dst_host])
            return path

        for spine in self.leaf_to_spines[src_leaf]:
            if spine in active_switches and spine in self.leaf_to_spines[dst_leaf]:
                path.extend([spine, dst_leaf, dst_host])
                return path
        
        return None
    
    def verify_connectivity(self, switch_set):
        """Verify all traffic demands can be satisfied with given switch set"""
        for src in self.traffic_matrix:
            for dst in self.traffic_matrix[src]:
                if self.traffic_matrix[src][dst] > 0:
                    path = self.find_path(src, dst, switch_set)
                    if not path:
                        return False
        return True
    
    def calculate_cost(self, switch_set):
        """
        Calculate cost (objective function) for a switch configuration.
        Cost = number of switches
        """
        return len(switch_set)
    
    def formal_optimization(self):
        """
        Model-based optimization: Find minimal switch set.
        Uses iterative constraint solving approach.
        """
        required_switches = set()
        
        active_hosts = set()
        for src in self.traffic_matrix:
            if any(self.traffic_matrix[src].values()):
                active_hosts.add(src)
        for flows in self.traffic_matrix.values():
            for dst in flows:
                if flows[dst] > 0:
                    active_hosts.add(dst)
        
        active_leaves = set()
        for host in active_hosts:
            leaf = self.host_to_leaf.get(host)
            if leaf:
                active_leaves.add(leaf)
                required_switches.add(leaf)
        
        
        if len(active_leaves) <= 1:
            spine_config = set()
        else:
            spine_config = None
            
            for num_spines in range(1, len(self.topo.SpineSwitchList) + 1):
                test_spines = set(self.topo.SpineSwitchList[:num_spines])
                test_switches = required_switches | test_spines
                
                if self.verify_connectivity(test_switches):
                    spine_config = test_spines
                    info(f"    Found connectivity with {num_spines} spine(s)\n")
                    break
            
            if not spine_config:
                info("    Warning: Using all spines\n")
                spine_config = set(self.topo.SpineSwitchList)
        
        required_switches.update(spine_config)
        
        original_size = len(required_switches)
        
        removable = []
        for switch in list(required_switches):
            if switch in active_leaves:
                continue
            
            test_switches = required_switches - {switch}
            if self.verify_connectivity(test_switches):
                removable.append(switch)
        
        if removable:
            for switch in removable:
                required_switches.remove(switch)
        else:
            info(f"Configuration is minimal (no redundancy)\n")
        
        final_size = len(required_switches)
        info(f"\n  Optimization result: {original_size} → {final_size} switches\n")
        
        required_links = set()
        
        for host in active_hosts:
            leaf = self.host_to_leaf.get(host)
            if leaf and leaf in required_switches:
                required_links.add((host, leaf))
        
        for leaf in active_leaves:
            for spine in self.leaf_to_spines[leaf]:
                if spine in required_switches:
                    required_links.add((leaf, spine))
        
        return required_switches, required_links
    
    def optimize_topology(self):
        """Main optimization routine"""
        total_traffic = sum(sum(flows.values()) for flows in self.traffic_matrix.values())
        info(f"\nTotal network traffic demand: {total_traffic:.3f} Gbps\n")
        
        required_switches, required_links = self.formal_optimization()
        
        total_switches = len(self.topo.SpineSwitchList) + len(self.topo.LeafSwitchList)
        powered_down = total_switches - len(required_switches)
        switch_reduction = (powered_down / total_switches) * 100 if total_switches > 0 else 0
        
        self.active_switches = required_switches
        self.active_links = required_links
        info(f"  Total switches:      {total_switches}\n")
        info(f"  Active switches:     {len(required_switches)}\n")
        info(f"  Powered down:        {powered_down}\n")
        info(f"  Switch reduction:    {switch_reduction:.1f}%\n")
        
        spine_active = sum(1 for s in self.topo.SpineSwitchList if s in required_switches)
        leaf_active = sum(1 for s in self.topo.LeafSwitchList if s in required_switches)
        
        info(f"\n  Layer Breakdown:\n")
        info(f"    Spine:  {spine_active}/{len(self.topo.SpineSwitchList)} active\n")
        info(f"    Leaf:   {leaf_active}/{len(self.topo.LeafSwitchList)} active\n")
        
        total_links = len(self.all_links)
        active_link_count = len(required_links)
        link_reduction = ((total_links - active_link_count) / total_links) * 100 if total_links > 0 else 0
        
        info(f"\n  Link Statistics:\n")
        info(f"    Total links:   {total_links}\n")
        info(f"    Active links:  {active_link_count}\n")
        info(f"    Link reduction: {link_reduction:.1f}%\n")
        
        self.visualize_topology()
        
        return self.active_switches
    
    def visualize_topology(self):
        """Visualize optimized topology state"""
        info("\nspine:\n")
        for spine in self.topo.SpineSwitchList:
            status = "ON " if spine in self.active_switches else "OFF"
            active_leaves = sum(1 for leaf in self.spine_to_leaves[spine] 
                              if leaf in self.active_switches)
            info(f"  {spine}: {status}  (connects to {active_leaves} active leaves)\n")
        
        info("\nleaf:\n")
        for leaf in self.topo.LeafSwitchList:
            status = "ON " if leaf in self.active_switches else "OFF"
            hosts = self.leaf_to_hosts[leaf]
            
            active_host_count = sum(1 for h in hosts if h in 
                                   [src for src in self.traffic_matrix.keys()] or
                                   h in [dst for flows in self.traffic_matrix.values() 
                                        for dst in flows.keys()])
            
            info(f"  {leaf}: {status}  <- {len(hosts)} hosts ({active_host_count} active)\n")
            info(f"        Hosts: {', '.join(hosts)}\n")
        
        info("\n" + "="*80 + "\n")


def run_topology():
    """Main function to run leaf-spine with formal optimization"""
    setLogLevel('info')
    
    num_spines = 4
    num_leaves = 6
    hosts_per_leaf = 4
    active_ratio = 0.4
    
    info(f"\nCreating Leaf-Spine Topology:\n")
    info(f"  Spines: {num_spines}\n")
    info(f"  Leaves: {num_leaves}\n")
    info(f"  Hosts per leaf: {hosts_per_leaf}\n")
    info(f"  Total hosts: {num_leaves * hosts_per_leaf}\n")
    info(f"  Active host ratio: {active_ratio*100:.0f}%\n\n")
    
    topo = LeafSpineTopo(num_spines=num_spines, 
                         num_leaves=num_leaves, 
                         hosts_per_leaf=hosts_per_leaf)
    
    net = Mininet(topo=topo, 
                  controller=RemoteController, 
                  switch=OVSKernelSwitch, 
                  link=TCLink, 
                  autoSetMacs=True)
    
    net.start()
    
    optimizer = LeafSpineFormalOptimizer(net, topo, active_ratio=active_ratio)
    optimizer.optimize_topology()
    
    CLI(net)
    net.stop()


if __name__ == '__main__':
    try:
        run_topology()
    except KeyboardInterrupt:
        from mininet.clean import cleanup
        cleanup()