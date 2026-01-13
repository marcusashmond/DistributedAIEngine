#ifndef NODE_H
#define NODE_H

#include <string>
#include <thread>
#include "Scheduler.h"
#include "Tensor.h"
#include "KVStore.h"
#include <vector>
#include <mutex>

class Node {
public:
    Node(int port, size_t numThreads, int nodeId = 0);
    ~Node();

    void startServer();
    void sendTask(const std::string& message);
    void broadcastTensor(const Tensor& tensor);
    Tensor receiveTensor();
    // Broadcast tensor to multiple destination ports
    void broadcastTensor(const Tensor& tensor, const std::vector<int>& destPorts);
    // Send tensor to a specific destination port (RPC-ready) - removed; use broadcast overload

private:
    int port;
    int serverSocket;
    // Protect access to clientSockets
    std::mutex clientsMutex;
    // Track connected client sockets
    std::vector<int> clientSockets;
    int nodeId;
    bool running;
    std::thread serverThread;

    Scheduler scheduler;
    KVStore kvStore;

    void serverLoop();
    void handleClient(int clientSocket);
    // Receive a tensor from a specific connected socket
    Tensor receiveTensor(int clientSocket);
    // Broadcast a tensor to all currently connected peers
    void broadcastToPeers(const Tensor& tensor);
    // Remove a dead socket safely
    void removeDeadSocket(int sock);
};

#endif
