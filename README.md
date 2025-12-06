# Mininet Docker Container

A ready-to-use Docker container with Mininet for SDN development and testing with Apple Silicon.

## Features

- **Mininet**: Full mininet installation for network emulation
- **Network Tools**: iperf, ping, curl, and other network utilities
- **Development Environment**: Python 3, pip, git, vim, nano
- **Apple Silicon Support**: Optimized for M1/M2/M3 Macs with full compatibility
- **Ready Examples**: Pre-built with topology examples included

## Quick Start

### 1. Build the Container

```bash
./docker-run.sh build
```

### 2. Run the Container

```bash
./docker-run.sh run
```

### 3. Test Mininet

Inside the container:

```bash
# Test basic mininet functionality
sudo mn --test pingall
```

### 4. Test Individual Optimization Algorithms

```bash
python3 projects/fattree/formal.py
python3 projects/fattree/greedy.py
python3 projects/fattree/topology-aware.py

python3 projects/jellyfish/formal.py
python3 projects/jellyfish/greedy.py
python3 projects/jellyfish/topology-aware.py

python3 projects/leaf-spine/formal.py
python3 projects/leaf-spine/greedy.py
python3 projects/leaf-spine/topology-aware.py
```

### 4. Reproduce Plots

Run the plotting script **inside the container**:

```bash
python3 projects/scripts/plot-download.py
```
Then, outside the container, copy the /app folder to your Desktop:
```bash
docker cp mininet-dev:/app ~/Desktop
```

## Available Commands

```bash
./docker-run.sh build     # Build Docker image
./docker-run.sh run       # Start container interactively
./docker-run.sh shell     # Enter running container
./docker-run.sh stop      # Stop container
./docker-run.sh status    # Show container status
./docker-run.sh logs      # View container logs
./docker-run.sh clean     # Remove container and image
./docker-run.sh help      # Show help
```
