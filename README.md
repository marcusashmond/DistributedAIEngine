# Distributed AI Runtime

## Project Highlights

✅ Complete C++17 distributed tensor operations system  
✅ ThreadPool-based concurrent execution  
✅ Fault-tolerant networking with dead socket removal  
✅ Disk-backed checkpointing (KVStore)  
✅ Neural execution graph with parallel node execution  
✅ TCP-based tensor broadcasting  
✅ Task scheduling system

## Architecture

```mermaid
graph TB
    subgraph NodeA["🔷 Node A (Port 5001)"]
        TP_A["🟢 ThreadPool<br/>(Worker Threads)"]
        KV_A["🟠 KVStore<br/>(In-Memory + Disk)"]
        GN_A["⚙️ Graph Nodes<br/>(Compute Tasks)"]
    end
    
    subgraph NodeB["🔷 Node B (Port 5002)"]
        TP_B["🟢 ThreadPool<br/>(Worker Threads)"]
        KV_B["🟠 KVStore<br/>(In-Memory + Disk)"]
        GN_B["⚙️ Graph Nodes<br/>(Compute Tasks)"]
    end
    
    subgraph NodeC["🔷 Node C (Port 5003)"]
        TP_C["🟢 ThreadPool<br/>(Worker Threads)"]
        KV_C["🟠 KVStore<br/>(In-Memory + Disk)"]
        GN_C["⚙️ Graph Nodes<br/>(Compute Tasks)"]
    end
    
    Disk_A[("💾 checkpoints/<br/>latest_tensor.chk")]
    Disk_B[("💾 checkpoints/<br/>latest_tensor.chk")]
    Disk_C[("💾 checkpoints/<br/>latest_tensor.chk")]
    
    NodeA -.->|"🔴 TCP Broadcast<br/>Tensor Data"| NodeB
    NodeB -.->|"🔴 TCP Broadcast<br/>Tensor Data"| NodeC
    NodeC -.->|"🔴 TCP Broadcast<br/>Tensor Data"| NodeA
    
    TP_A -->|Executes| GN_A
    TP_B -->|Executes| GN_B
    TP_C -->|Executes| GN_C
    
    KV_A -->|"saveToDisk()"| Disk_A
    KV_B -->|"saveToDisk()"| Disk_B
    KV_C -->|"saveToDisk()"| Disk_C
    
    Disk_A -.->|"loadFromDisk()<br/>(on startup)"| KV_A
    Disk_B -.->|"loadFromDisk()<br/>(on startup)"| KV_B
    Disk_C -.->|"loadFromDisk()<br/>(on startup)"| KV_C
    
    style NodeA fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style NodeB fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    style NodeC fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    
    style TP_A fill:#50C878,stroke:#2D7A4A,stroke-width:2px
    style TP_B fill:#50C878,stroke:#2D7A4A,stroke-width:2px
    style TP_C fill:#50C878,stroke:#2D7A4A,stroke-width:2px
    
    style KV_A fill:#FF8C42,stroke:#CC6A2F,stroke-width:2px
    style KV_B fill:#FF8C42,stroke:#CC6A2F,stroke-width:2px
    style KV_C fill:#FF8C42,stroke:#CC6A2F,stroke-width:2px
```

**Components:**

- **🔷 Node (Blue)**: Distributed compute node with TCP server
- **🟢 ThreadPool (Green)**: Concurrent task execution with worker threads
- **🟠 KVStore (Orange)**: In-memory tensor storage with disk persistence
- **⚙️ Graph Nodes**: Execution graph for neural computation
- **🔴 Broadcast (Red arrows)**: TCP-based tensor distribution between nodes
- **💾 Checkpoints**: Disk-backed tensor serialization

## Overview

A C++ distributed AI runtime prototype supporting:

- Multi-node tensor broadcast
- Thread-pool task scheduling
- Disk-backed checkpointing
- Minimal neural execution graph

This project demonstrates distributed systems engineering, concurrency, and ML infrastructure in a single, internship-ready repository.

## Screenshots

### Tensor Broadcast & Network Communication

![Broadcast Output](screenshots/broadcast_output.png%20.png)
*Multi-threaded tensor broadcasting between nodes over TCP with fault-tolerant socket management*

### Graph Execution

![Graph Execution](screenshots/graph_execution.png%20.png)
*Parallel execution of computation graph nodes via ThreadPool with thread IDs displayed*

### Checkpoint System

![Checkpoint File](screenshots/checkpoint_file.png%20.png)
*Disk-backed tensor checkpointing with binary serialization (TENS format)*

