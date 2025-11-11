#!/usr/bin/env python3
import math, random, csv, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from formal import FormalOptimizer, FatTree
from greedy import GreedyBinPackingOptimizer
from topology_aware import ElasticTreeOptimizer
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info


# ============================================================
# PARAMETERS
# ============================================================
TOPOLOGIES = ['fattree']
LOAD_LEVELS = [0.1, 0.3, 0.5, 0.7, 1.0]
LOCALITIES = ['near','mid','far']
REPEATS = 5  # number of random traffic matrices per combo
RESULTS_CSV = "replication_results_multi.csv"
os.makedirs("figures_real", exist_ok=True)

# ============================================================
# UTILS
# ============================================================
def set_locality(optimizer, locality):
    """Randomize traffic matrix locality pattern safely for any topology size."""
    if not hasattr(optimizer, "traffic_matrix"):
        return

    hosts = list(optimizer.traffic_matrix.keys())
    pods = optimizer.topo.pod
    end = max(1, int(pods / 2))

    # Dynamically allocate hosts into pod groups
    host_groups = [[] for _ in range(pods)]
    for i, h in enumerate(hosts):
        pod_index = min(i // end, pods - 1)
        host_groups[pod_index].append(h)

    tm = optimizer.traffic_matrix
    for src in tm:
        for dst in tm[src]:
            tm[src][dst] = 0.0

    for src in hosts:
        for dst in hosts:
            if src == dst:
                continue
            if locality == 'near':  # same edge
                if optimizer.host_to_edge[src] == optimizer.host_to_edge[dst]:
                    tm[src][dst] = random.uniform(0.01, 0.3)
            elif locality == 'mid':  # same pod
                if optimizer.edge_to_pod[optimizer.host_to_edge[src]] == optimizer.edge_to_pod[optimizer.host_to_edge[dst]]:
                    tm[src][dst] = random.uniform(0.05, 0.5)
            else:  # far, across pods
                if optimizer.edge_to_pod[optimizer.host_to_edge[src]] != optimizer.edge_to_pod[optimizer.host_to_edge[dst]]:
                    tm[src][dst] = random.uniform(0.05, 0.5)


# ============================================================
# EXPERIMENTS
# ============================================================
def run_experiment():
    setLogLevel('info')
    with open(RESULTS_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Topology","Algorithm","Locality","Load","Repeat","PowerSavings"])

        for topo_name in TOPOLOGIES:
            topo = FatTree(4)
            for load in LOAD_LEVELS:
                for locality in LOCALITIES:
                    for r in range(REPEATS):
                        net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch,
                                      link=TCLink, autoSetMacs=True)
                        net.start()
                        info(f"\n>>> Run {r+1}/{REPEATS} Load={load}, Locality={locality}\n")

                        # FORMAL
                        fm = FormalOptimizer(net, topo, active_ratio=load)
                        set_locality(fm, locality)
                        fm.optimize_topology()
                        total_sw = len(topo.CoreSwitchList)+len(topo.AggSwitchList)+len(topo.EdgeSwitchList)
                        formal_savings = (total_sw - len(fm.active_switches)) / total_sw * 100
                        writer.writerow([topo_name,"Formal",locality,load,r,formal_savings])

                        # GREEDY
                        gr = GreedyBinPackingOptimizer(net, topo)
                        gr.optimize_topology()
                        greedy_savings = (total_sw - len(gr.active_switches)) / total_sw * 100
                        writer.writerow([topo_name,"Greedy",locality,load,r,greedy_savings])

                        # ELASTIC TREE
                        ta = ElasticTreeOptimizer(net, topo, active_ratio=load)
                        set_locality(ta, locality)
                        ta.optimize_topology()
                        ta_savings = (total_sw - len(ta.active_switches)) / total_sw * 100
                        writer.writerow([topo_name,"ElasticTree",locality,load,r,ta_savings])

                        net.stop()

# ============================================================
# PLOTTING
# ============================================================
def plot_results():
    df = pd.read_csv(RESULTS_CSV)
    df["PowerUsed"] = 100 - df["PowerSavings"]

    # === Figure 7: average power vs load (same as before)
    plt.figure(figsize=(6.5,4.5))
    for locality in LOCALITIES:
        subset = df[(df["Algorithm"]=="Formal") & (df["Locality"]==locality)]
        mean = subset.groupby("Load")["PowerUsed"].mean()
        plt.plot(mean.index, mean.values, marker='o', label=locality.capitalize(), linewidth=2)
    plt.xlabel("Traffic Load (fraction)")
    plt.ylabel("% Original Network Power")
    plt.title("Figure 7 – Power Consumption vs Load (Fat-Tree)")
    plt.grid(True, ls=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_real/figure7_fattree_real.png", dpi=300)

    # === Figure 8: true scatter, not synthetic ===
    plt.figure(figsize=(6.5,4.5))
    greedy = df[df["Algorithm"]=="Greedy"]
    topo = df[df["Algorithm"]=="ElasticTree"]
    plt.scatter(greedy["Load"], greedy["PowerUsed"], color='blue', s=25, alpha=0.6, label="Greedy")
    plt.scatter(topo["Load"], topo["PowerUsed"], color='red', s=25, alpha=0.6, label="Topology-Aware")
    plt.xlabel("Average Utilization (fraction)")
    plt.ylabel("% Original Network Power")
    plt.title("Figure 8 – Random Traffic Scatter (Real Data)")
    plt.grid(True, ls=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_real/figure8_scatter_real.png", dpi=300)

    # === Figure 9: Diurnal average ===
    hours = list(range(24))
    df_avg = df.groupby("Load")["PowerUsed"].mean()
    loads = np.linspace(0.1, 1.0, len(hours))
    interp = np.interp(loads, df_avg.index, df_avg.values)
    plt.figure(figsize=(6.5,4.5))
    plt.plot(hours, interp, 'o-', color='green', label="Average Power Usage")
    plt.xlabel("Time (hours)")
    plt.ylabel("% Original Network Power")
    plt.title("Figure 9 – Diurnal Variation (From Real Samples)")
    plt.grid(True, ls=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures_real/figure9_diurnal_real.png", dpi=300)

    print("✅ Figures saved in ./figures_real/")

# ============================================================
if __name__ == "__main__":
    try:
        run_experiment()
        plot_results()
    except KeyboardInterrupt:
        from mininet.clean import cleanup
        cleanup()
