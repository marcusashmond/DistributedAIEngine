# DistributedAIEngine - AI Coding Agent Instructions

## Architecture Overview

This is a C++17 distributed computing system for parallel tensor operations. The architecture follows a peer-to-peer model with three core subsystems:

1. **Node System** ([src/Node.cpp](../src/Node.cpp), [include/Node.h](../include/Node.h))
   - Each node runs a TCP server that accepts connections and maintains a list of connected client sockets
   - Broadcasting tensors uses the tracked `clientSockets` vector (protected by `clientsMutex`)
   - Two broadcast modes: (1) to all connected peers via `broadcastTensor(tensor)`, (2) RPC-style to specific ports via `broadcastTensor(tensor, destPorts)`
   - Incoming tensors trigger computation tasks submitted to the scheduler

2. **ThreadPool + Scheduler** ([src/ThreadPool.cpp](../src/ThreadPool.cpp), [src/Scheduler.cpp](../src/Scheduler.cpp))
   - ThreadPool implements standard work-stealing pattern with condition variables
   - Scheduler wraps ThreadPool, converting `Task` structs into lambda functions
   - Tasks are defined in [include/Task.h](../include/Task.h) with type (COMPUTE/IO), name, tensor, and work function

3. **Tensor Serialization** ([src/Tensor.cpp](../src/Tensor.cpp))
   - Binary format: 8-byte header (`TENS` + version + dtype + reserved), then little-endian shape dimensions, then raw float32 data
   - Network protocol: send uint64_t length prefix (host endianness), then binary payload
   - Supports both text (`serialize()`/`deserialize()`) and binary serialization

## Build & Test Workflows

**Building the project:**
```bash
make                    # Builds to build/DistributedAIEngine
```

**Testing:**
```bash
# ThreadPool test (in tests/test_threadpool.cpp)
g++ -std=c++17 -pthread -Iinclude tests/test_threadpool.cpp src/ThreadPool.cpp -o build/test_threadpool
./build/test_threadpool
```

**Running the demo:**
```bash
./build/DistributedAIEngine    # Starts nodeA on port 5001, spawns 2 client threads
```

The Makefile uses `g++` with `-std=c++17 -pthread -Iinclude -Wall -Wextra -O2`. All headers are in `include/`, implementations in `src/`.

## Key Conventions & Patterns

- **Socket Management**: Always protect `clientSockets` access with `clientsMutex`. Sockets are added in `serverLoop()` and removed in `handleClient()` after closure.

- **Tensor Network Protocol**: Fixed 8-byte `uint64_t` length prefix (host byte order), then binary payload. Receiving side uses `MSG_WAITALL` for length, then loops for full payload.

- **Error Handling**: Socket operations print to `std::cerr` and continue/return. Tensor deserialization throws `std::runtime_error` on invalid data.

- **Thread Safety**: Only `clientSockets` requires mutex protection. Tasks execute independently in ThreadPool workers.

- **Memory Management**: Sockets explicitly closed with `close()`. ThreadPool joins all workers in destructor. No manual memory allocation—uses STL containers.

## Common Implementation Patterns

**Adding a new Task type:**
1. Add enum value to `TaskType` in [include/Task.h](../include/Task.h)
2. Define task creation in Node's `handleClient()` or caller code
3. Set `task.work` lambda with required computation

**Adding a new Node method:**
1. Declare in [include/Node.h](../include/Node.h) (check if mutex protection needed)
2. Implement in [src/Node.cpp](../src/Node.cpp)
3. For network operations, follow the length-prefix protocol pattern

**Extending Tensor serialization:**
- Binary format is versioned (currently v1). Increment version byte for incompatible changes.
- New dtypes require adding to dtype enum (byte 5) and branching in deserialize logic.

## Project-Specific Rules

- Include headers with quotes, not angle brackets: `#include "Node.h"`
- Use POSIX sockets directly (no Boost/Asio abstraction)
- Prefer detached threads for per-client handlers; join only for lifecycle threads (serverThread)
- Node ID is tracked but not currently used in networking—reserved for future distributed coordination
- CMakeLists.txt is empty; the Makefile is the canonical build system
