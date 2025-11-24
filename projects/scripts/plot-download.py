import subprocess
import glob
import shutil
import os

scripts = [
    "projects/scripts/fat-tree/fattree_formal_plots.py",
    "projects/scripts/fat-tree/fattree_greedy_plots.py",
    "projects/scripts/fat-tree/fattree_topology_aware_plots.py",

    "projects/scripts/jellyfish/jellyfish_formal_optimization.py",
    "projects/scripts/jellyfish/jellyfish_greedy_optimization.py",
    "projects/scripts/jellyfish/jellyfish_topology_aware_optimization.py",

    "projects/scripts/leaf_spine/leafspine_formal_analysis_plots.py",
    "projects/scripts/leaf_spine/leafspine_greedy_analysis_plots.py",
    "projects/scripts/leaf_spine/leafspine_topology_aware_analysis.py"
]

for script in scripts:
    print(f"\n Running: {script}")
    subprocess.run(["python3", script])

DEST_DIR = "/app"
os.makedirs(DEST_DIR, exist_ok=True)
png_files = glob.glob("**/*.png", recursive=True)
for file in png_files:
    filename = os.path.basename(file)
    dest_path = os.path.join(DEST_DIR, filename)
    shutil.move(file, dest_path)