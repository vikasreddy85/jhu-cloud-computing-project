#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict

try:
    from projects.fattree.formal import FatTree, FormalOptimizer
except ImportError as e:
    print(f"ERROR: Could not import from formal_optimization.py")
    print(f"Error details: {e}")
    exit(1)

K = 4
NUM_CORE = int((K/2)**2) 
NUM_AGG = int(K*K/2)
NUM_EDGE = int(K*K/2)
HOSTS_PER_EDGE = int(K/2)
TOTAL_SWITCHES = NUM_CORE + NUM_AGG + NUM_EDGE
POWER_PER_SWITCH = 10
BASELINE_POWER = TOTAL_SWITCHES * POWER_PER_SWITCH


class MockMininet:
    def __init__(self, topo):
        self.topo = topo
        self.hosts = [MockHost(h) for h in topo.HostList]
        
class MockHost:
    def __init__(self, name):
        self.name = name


def generate_traffic_matrix(topo, active_ratio, pattern_type, seed):
    random.seed(seed)
    
    traffic_matrix = defaultdict(lambda: defaultdict(float))
    all_hosts = topo.HostList    
    num_active = max(1, int(len(all_hosts) * active_ratio))
    active_hosts = random.sample(all_hosts, num_active)
    
    host_to_edge = {}
    edge_to_pod = {}
    pod_to_hosts = defaultdict(list)
    
    density = int(topo.density)
    end = int(topo.pod / 2)
    
    for i, host in enumerate(topo.HostList):
        edge_idx = i // density
        if edge_idx < len(topo.EdgeSwitchList):
            edge = topo.EdgeSwitchList[edge_idx]
            host_to_edge[host] = edge
            
            pod_idx = edge_idx // end
            edge_to_pod[edge] = pod_idx
            pod_to_hosts[pod_idx].append(host)
    
    for src in active_hosts:
        src_edge = host_to_edge.get(src)
        src_pod = edge_to_pod.get(src_edge)
        
        same_pod_hosts = [h for h in pod_to_hosts[src_pod] 
                         if h in active_hosts and h != src]
        other_pod_hosts = [h for h in active_hosts 
                          if h != src and h not in same_pod_hosts]
        
        for dst in active_hosts:
            if src != dst:
                dst_edge = host_to_edge.get(dst)
                dst_pod = edge_to_pod.get(dst_edge)
                
                if pattern_type == 'near':
                    if src_pod == dst_pod:
                        traffic_matrix[src][dst] = random.uniform(0.1, 0.5)
                    elif random.random() < 0.2:  # 20% inter-pod
                        traffic_matrix[src][dst] = random.uniform(0.01, 0.1)
                
                elif pattern_type == 'far':
                    if src_pod != dst_pod:
                        traffic_matrix[src][dst] = random.uniform(0.1, 0.5)
                    elif random.random() < 0.2:  # 20% intra-pod
                        traffic_matrix[src][dst] = random.uniform(0.01, 0.1)
                
                else:
                    traffic_matrix[src][dst] = random.uniform(0.01, 0.5)
    
    return traffic_matrix


def run_single_formal_optimization(active_ratio, traffic_pattern, seed):
    topo = FatTree(k=K)
    net = MockMininet(topo)
    
    optimizer = FormalOptimizer(net, topo, active_ratio=active_ratio)
    optimizer.traffic_matrix = generate_traffic_matrix(topo, active_ratio, traffic_pattern, seed)
    
    required_switches, required_links = optimizer.formal_optimization()
    
    power_consumption = len(required_switches) * POWER_PER_SWITCH
    num_active_switches = len(required_switches)
    num_active_links = len(required_links)
    
    active_cores = sum(1 for s in topo.CoreSwitchList if s in required_switches)
    active_aggs = sum(1 for s in topo.AggSwitchList if s in required_switches)
    active_edges = sum(1 for s in topo.EdgeSwitchList if s in required_switches)
    
    return (power_consumption, num_active_switches, num_active_links, 
            active_cores, active_aggs, active_edges)


def plot1_power_vs_load():
    print("\nGenerating Plot 1: Power vs Load (Near/Mid/Far patterns)...")
    
    loads = np.linspace(0.1, 1.0, 10)
    
    power_near = []
    power_mid = []
    power_far = []
    
    for i, load in enumerate(loads):
        print(f"  Processing load {load*100:.0f}% ({i+1}/{len(loads)})...")
        
        seed = int(load * 1000)
        
        p_near, _, _, _, _, _ = run_single_formal_optimization(load, 'near', seed)
        power_near.append(p_near)
        
        p_mid, _, _, _, _, _ = run_single_formal_optimization(load, 'balanced', seed + 1)
        power_mid.append(p_mid)
        
        p_far, _, _, _, _, _ = run_single_formal_optimization(load, 'far', seed + 2)
        power_far.append(p_far)
    
    baseline = [BASELINE_POWER] * len(loads)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loads * 100, power_near, 'g-', marker='o', label='Near (Intra-Pod)', 
             linewidth=2.5, markersize=8)
    plt.plot(loads * 100, power_mid, 'b-', marker='s', label='Mid (Balanced)', 
             linewidth=2.5, markersize=8)
    plt.plot(loads * 100, power_far, 'r-', marker='^', label='Far (Inter-Pod)', 
             linewidth=2.5, markersize=8)
    plt.plot(loads * 100, baseline, 'k--', label='Baseline (No Optimization)', 
             linewidth=2.5)
    
    plt.xlabel('Network Load (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Power Consumption (W)', fontsize=13, fontweight='bold')
    plt.title('Fat-Tree Formal Optimization: Power vs Load\n', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(5, 105)
    plt.ylim(0, BASELINE_POWER * 1.1)
    
    avg_savings_near = (1 - np.mean(power_near) / BASELINE_POWER) * 100
    avg_savings_mid = (1 - np.mean(power_mid) / BASELINE_POWER) * 100
    avg_savings_far = (1 - np.mean(power_far) / BASELINE_POWER) * 100
    
    stats_text = f'Avg Power Savings:\nNear:  {avg_savings_near:.1f}%\nMid:   {avg_savings_mid:.1f}%\nFar:   {avg_savings_far:.1f}%'
    plt.text(0.98, 0.05, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fattree_formal_plot1_power_vs_load.png', dpi=300, bbox_inches='tight')


def plot2_random_traffic_scatter():
    print("\nGenerating Plot 2: Random Traffic Scatter...")
    
    num_samples = 100
    
    loads = []
    powers = []
    patterns = []
    core_counts = []
    
    pattern_types = ['near', 'balanced', 'far']
    pattern_colors = {'near': 'green', 'balanced': 'blue', 'far': 'red'}
    
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{num_samples}...")
        
        load = np.random.uniform(0.1, 1.0)
        pattern = random.choice(pattern_types)
        
        power, _, _, core_count, _, _ = run_single_formal_optimization(load, pattern, seed=i*100)
        
        loads.append(load * 100)
        powers.append(power)
        patterns.append(pattern)
        core_counts.append(core_count)
    
    plt.figure(figsize=(10, 6))
    
    for pattern in pattern_types:
        pattern_loads = [loads[i] for i in range(len(loads)) if patterns[i] == pattern]
        pattern_powers = [powers[i] for i in range(len(powers)) if patterns[i] == pattern]
        plt.scatter(pattern_loads, pattern_powers, 
                   c=pattern_colors[pattern], 
                   label=f'{pattern.capitalize()} Traffic',
                   alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    plt.axhline(y=BASELINE_POWER, color='black', linestyle='--', linewidth=2.5, label='Baseline')
    
    plt.xlabel('Network Load (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Power Consumption (W)', fontsize=13, fontweight='bold')
    plt.title('Fat-Tree Random Traffic Scenarios: Load vs Power\n', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(5, 105)
    plt.ylim(0, BASELINE_POWER * 1.1)
    
    avg_power = np.mean(powers)
    avg_savings = (1 - avg_power / BASELINE_POWER) * 100
    avg_cores = np.mean(core_counts)
    stats_text = f'Overall Average:\nPower: {avg_power:.1f}W\nSavings: {avg_savings:.1f}%\nAvg Cores: {avg_cores:.1f}/{NUM_CORE}\nSamples: {num_samples}'
    plt.text(0.98, 0.05, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fattree_formal_plot2_random_traffic_scatter.png', dpi=300, bbox_inches='tight')

def plot3_diurnal_variation():
    print("\nGenerating Plot 3: Diurnal Variation (24-hour pattern)...")
    
    hours = np.arange(0, 24, 0.5)
    
    load_variation = []
    
    for hour in hours:
        if 3 <= hour < 6:
            load = 0.25 + np.random.uniform(-0.05, 0.05)
        elif 9 <= hour < 17:
            load = 0.85 + np.random.uniform(-0.1, 0.1)
        elif 6 <= hour < 9:
            load = 0.25 + (0.85 - 0.25) * (hour - 6) / 3 + np.random.uniform(-0.05, 0.05)
        elif 17 <= hour < 21:
            load = 0.85 - (0.85 - 0.25) * (hour - 17) / 4 + np.random.uniform(-0.05, 0.05)
        else:
            load = 0.25 + np.random.uniform(-0.05, 0.1)
        
        load_variation.append(np.clip(load, 0.15, 1.0))
    
    optimized_power = []
    for i, load in enumerate(load_variation):
        if i % 10 == 0:
            print(f"  Processing hour {hours[i]:.1f} ({i+1}/{len(hours)})...")
        
        p, _, _, _, _, _ = run_single_formal_optimization(load, 'balanced', seed=int(i*100))
        optimized_power.append(p)
    
    baseline_power = [BASELINE_POWER] * len(hours)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(hours, baseline_power, 'k--', linewidth=2.5, label='Baseline (No Optimization)')
    plt.plot(hours, optimized_power, 'b-', linewidth=2.5, label='Formal Optimization', 
             marker='o', markersize=4)
    
    plt.fill_between(hours, optimized_power, baseline_power, 
                     alpha=0.3, color='green', label='Energy Saved')
    
    plt.xlabel('Time of Day (hours)', fontsize=13, fontweight='bold')
    plt.ylabel('Power Consumption (W)', fontsize=13, fontweight='bold')
    plt.title('Fat-Tree Diurnal Variation: 24-Hour Power Consumption\n', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 3))
    
    time_labels = ['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm', '12am']
    plt.gca().set_xticklabels(time_labels)
    
    baseline_energy = BASELINE_POWER * 24
    optimized_energy = np.trapz(optimized_power, hours)
    energy_saved = baseline_energy - optimized_energy
    savings_percent = (energy_saved / baseline_energy) * 100
    
    stats_text = f'24-Hour Statistics:\nBaseline: {baseline_energy:.0f} Wh\nOptimized: {optimized_energy:.0f} Wh\nSaved: {energy_saved:.0f} Wh ({savings_percent:.1f}%)'
    plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fattree_formal_plot3_diurnal_variation.png', dpi=300, bbox_inches='tight')


def main():
    random.seed(42)
    np.random.seed(42)
    
    plot1_power_vs_load()
    plot2_random_traffic_scatter()
    plot3_diurnal_variation()

if __name__ == '__main__':
    main()