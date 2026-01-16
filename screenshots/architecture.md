# Architecture Diagram

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

## Components

- **🔷 Node (Blue)**: Distributed compute node with TCP server
- **🟢 ThreadPool (Green)**: Concurrent task execution with worker threads
- **🟠 KVStore (Orange)**: In-memory tensor storage with disk persistence
- **⚙️ Graph Nodes**: Execution graph for neural computation
- **🔴 Broadcast (Red arrows)**: TCP-based tensor distribution between nodes
- **💾 Checkpoints**: Disk-backed tensor serialization

## Data Flow

1. Client threads send tensors to Node's TCP server (port 5001)
2. Node stores tensor in KVStore and calls `saveToDisk()`
3. ThreadPool executes compute tasks on received tensors
4. Graph nodes run concurrently via ThreadPool
5. Nodes broadcast tensors to connected peers over TCP
6. On startup, nodes restore state via `loadFromDisk()`
