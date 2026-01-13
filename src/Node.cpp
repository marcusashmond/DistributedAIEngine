#include "Node.h"
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <chrono>
#include "Tensor.h"
#include <algorithm>

Node::Node(int port, size_t numThreads, int nodeId)
        : port(port),
            serverSocket(-1),
            nodeId(nodeId),
            running(false),
            scheduler(numThreads) {
    // Attempt to restore checkpoint on startup
    kvStore.loadFromDisk("latest_tensor");
}

Node::~Node() {
    running = false;
    if (serverSocket != -1) {
        close(serverSocket);
    }
    if (serverThread.joinable()) {
        serverThread.join();
    }
}

void Node::startServer() {
    running = true;
    serverThread = std::thread(&Node::serverLoop, this);
}

void Node::serverLoop() {
    // NOTE: Server runs indefinitely; clean shutdown not yet implemented.
    // accept() blocks — this is acceptable for the prototype. We'll add
    // signal handling, control messages, or lifecycle management later.
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        std::cerr << "Failed to create socket\n";
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);

    int opt = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Bind failed\n";
        close(serverSocket);
        serverSocket = -1;
        return;
    }

    if (listen(serverSocket, 5) < 0) {
        std::cerr << "Listen failed\n";
        close(serverSocket);
        serverSocket = -1;
        return;
    }

    std::cout << "Node listening on port " << port << std::endl;

    while (true) {
        int clientSocket = accept(serverSocket, nullptr, nullptr);
        if (clientSocket >= 0) {
            // Track client socket and dispatch handler thread
            {
                std::lock_guard<std::mutex> lg(clientsMutex);
                clientSockets.push_back(clientSocket);
            }

            std::thread(&Node::handleClient, this, clientSocket).detach();
        } else {
            if (!running) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void Node::handleClient(int clientSocket) {
    try {
        Tensor received = receiveTensor(clientSocket);

        kvStore.put("latest_tensor", received);
        kvStore.saveToDisk("latest_tensor");

        std::cout << "DEBUG: Submitting task for tensor of size " << received.size() << std::endl;

        Task task;
        task.type = TaskType::COMPUTE;
        task.name = "TensorCompute";
        task.tensor = received;

        task.work = [this](const Tensor& t) {
            Tensor restored;
            if (kvStore.get("latest_tensor", restored)) {
                float sum = 0.0f;
                for (size_t i = 0; i < restored.size(); i++) {
                    sum += restored[i];
                }
                std::cout << "KVStore tensor sum: " << sum << std::endl;
                std::cout.flush();
            } else {
                std::cout << "Failed to retrieve from KVStore" << std::endl;
            }
        };

        scheduler.submitTask(task);
    } catch (const std::exception& e) {
        std::cerr << "receiveTensor failed: " << e.what() << std::endl;
        removeDeadSocket(clientSocket);
        return;
    }

    close(clientSocket);
    // remove from clientSockets tracking
    {
        std::lock_guard<std::mutex> lg(clientsMutex);
        auto it = std::find(clientSockets.begin(), clientSockets.end(), clientSocket);
        if (it != clientSockets.end()) clientSockets.erase(it);
    }
}

void Node::removeDeadSocket(int sock) {
    std::lock_guard<std::mutex> lock(clientsMutex);

    auto it = std::find(clientSockets.begin(), clientSockets.end(), sock);
    if (it != clientSockets.end()) {
        close(sock);
        clientSockets.erase(it);
    }
}

void Node::sendTask(const std::string& message) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Socket create failed\n";
        return;
    }

    sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Connection failed\n";
        close(sock);
        return;
    }

    // Send 8-byte big-endian length prefix followed by payload
    uint64_t len = message.size();
    uint8_t lenbuf[8];
    for (int i = 7; i >= 0; --i) {
        lenbuf[i] = static_cast<uint8_t>(len & 0xFF);
        len >>= 8;
    }

    size_t sent = 0;
    while (sent < 8) {
        ssize_t s = ::send(sock, lenbuf + sent, 8 - sent, 0);
        if (s <= 0) break;
        sent += static_cast<size_t>(s);
    }

    sent = 0;
    const char* data = message.data();
    size_t tosend = message.size();
    while (sent < tosend) {
        ssize_t s = ::send(sock, data + sent, tosend - sent, 0);
        if (s <= 0) break;
        sent += static_cast<size_t>(s);
    }

    close(sock);
}

void Node::broadcastTensor(const Tensor& tensor) {
    auto buffer = tensor.serializeBinary();
    uint64_t size = static_cast<uint64_t>(buffer.size());

    std::vector<int> socketsCopy;
    {
        std::lock_guard<std::mutex> lock(clientsMutex);
        socketsCopy = clientSockets;
    }

    for (int sock : socketsCopy) {
        ssize_t sent = send(sock, &size, sizeof(uint64_t), 0);
        if (sent <= 0) {
            removeDeadSocket(sock);
            continue;
        }

        sent = send(sock, buffer.data(), size, 0);
        if (sent <= 0) {
            removeDeadSocket(sock);
        }
    }
}

// RPC-style sendTensor removed — broadcasting uses tracked clientSockets now.

void Node::broadcastTensor(const Tensor& tensor, const std::vector<int>& destPorts) {
    for (int p : destPorts) {
        try {
            // RPC-style send to destPort
            std::vector<char> data = tensor.serializeBinary();

            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                std::cerr << "Socket create failed\n";
                continue;
            }

            sockaddr_in serverAddr{};
            serverAddr.sin_family = AF_INET;
            serverAddr.sin_port = htons(p);
            serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

            if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
                std::cerr << "Connection failed to port " << p << "\n";
                close(sock);
                continue;
            }

            uint64_t len = data.size();
            uint8_t lenbuf[8];
            for (int i = 7; i >= 0; --i) {
                lenbuf[i] = static_cast<uint8_t>(len & 0xFF);
                len >>= 8;
            }

            size_t sent = 0;
            while (sent < 8) {
                ssize_t s = ::send(sock, lenbuf + sent, 8 - sent, 0);
                if (s <= 0) break;
                sent += static_cast<size_t>(s);
            }

            sent = 0;
            const char* bytes = data.data();
            size_t tosend = data.size();
            while (sent < tosend) {
                ssize_t s = ::send(sock, bytes + sent, tosend - sent, 0);
                if (s <= 0) break;
                sent += static_cast<size_t>(s);
            }

            close(sock);
        } catch (...) {
            std::cerr << "broadcast: failed to send to port " << p << std::endl;
        }
    }
}

Tensor Node::receiveTensor(int clientSocket) {
    // Read 8-byte big-endian length prefix
    uint8_t lenbuf[8];
    size_t have = 0;
    while (have < 8) {
        ssize_t r = ::read(clientSocket, lenbuf + have, 8 - have);
        if (r <= 0) throw std::runtime_error("Failed reading length prefix");
        have += static_cast<size_t>(r);
    }

    uint64_t len = 0;
    for (int i = 0; i < 8; ++i) {
        len = (len << 8) | static_cast<uint8_t>(lenbuf[i]);
    }

    std::vector<char> payload;
    payload.resize(len);
    size_t received = 0;
    while (received < len) {
        ssize_t r = ::read(clientSocket, payload.data() + received, static_cast<size_t>(len) - received);
        if (r <= 0) throw std::runtime_error("Failed reading tensor payload");
        received += static_cast<size_t>(r);
    }

    return Tensor::deserializeBinary(payload);
}

void Node::broadcastToPeers(const Tensor& tensor) {
    broadcastTensor(tensor);
}
