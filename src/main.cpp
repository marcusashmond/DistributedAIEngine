#include "Node.h"
#include "Tensor.h"
#include "Graph.h"
#include "GraphNode.h"
#include "ThreadPool.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>

int main() {
    Tensor orig({2, 3});  // 2x3 tensor

    for (size_t i = 0; i < orig.size(); i++) {
        orig[i] = static_cast<float>(i);
    }

    std::cout << "Tensor size: " << orig.size() << std::endl;
    std::cout << "Tensor values: ";
    for (size_t i = 0; i < orig.size(); i++) {
        std::cout << orig[i] << " ";
    }
    std::cout << std::endl;

    // Test text-based serialization/deserialization
    std::string serialized = orig.serialize();
    Tensor restored = Tensor::deserialize(serialized);

    std::cout << "Restored tensor values: ";
    for (size_t i = 0; i < restored.size(); i++) {
        std::cout << restored[i] << " ";
    }
    std::cout << std::endl;

    // Test binary serialization/deserialization
    auto binary = orig.serializeBinary();
    Tensor restoredBinary = Tensor::deserializeBinary(binary);

    std::cout << "Binary restored values: ";
    for (size_t i = 0; i < restoredBinary.size(); i++) {
        std::cout << restoredBinary[i] << " ";
    }
    std::cout << std::endl;

    // Set up nodeA as the sender. Nodes B and C will be receivers (clients).
    Node nodeA(5001, 4, 1);
    nodeA.startServer();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Demonstrate broadcasting a tensor from nodeA to two receiving clients
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Start two receiver threads (Nodes B and C) that connect to nodeA and wait for a tensor
    auto receiver = [&](int idx) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { std::cerr << "client socket create failed\n"; return; }

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(5001);
        serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "client connect failed\n";
            close(sock);
            return;
        }

        // Create and send a tensor TO the server
        Tensor clientTensor({2, 3});
        for (size_t i = 0; i < clientTensor.size(); i++) {
            clientTensor[i] = static_cast<float>(i);
        }

        std::vector<char> data = clientTensor.serializeBinary();
        uint64_t len = data.size();

        // Send length prefix (big-endian)
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

        // Send payload
        sent = 0;
        const char* bytes = data.data();
        size_t tosend = data.size();
        while (sent < tosend) {
            ssize_t s = ::send(sock, bytes + sent, tosend - sent, 0);
            if (s <= 0) break;
            sent += static_cast<size_t>(s);
        }

        std::cout << "Client " << idx << " sent tensor to server" << std::endl;

        close(sock);
    };

    std::thread r1(receiver, 1);
    std::thread r2(receiver, 2);

    // Wait for clients to send tensors
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    r1.join();
    r2.join();

    // Wait for tasks to complete and print
    std::cout << "\n=== Waiting for tasks to complete ===" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Neural execution graph demo
    std::cout << "\n=== Execution Graph Demo ===" << std::endl;

    // Create a thread pool for graph execution
    ThreadPool pool(4);

    auto nodeA_graph = std::make_shared<GraphNode>();
    nodeA_graph->name = "Input";
    nodeA_graph->tensor = Tensor({2,3});
    for (size_t i=0; i<nodeA_graph->tensor.size(); i++) nodeA_graph->tensor[i] = i;

    auto nodeB_graph = std::make_shared<GraphNode>();
    nodeB_graph->name = "Double";
    nodeB_graph->inputs.push_back(nodeA_graph);
    nodeB_graph->tensor = Tensor({2,3});
    nodeB_graph->operation = [nodeB_graph]() {
        for (auto& input : nodeB_graph->inputs) {
            for (size_t i=0; i<input->tensor.size(); i++) {
                nodeB_graph->tensor[i] = input->tensor[i] * 2;
            }
        }
    };

    Graph graph;
    graph.nodes.push_back(nodeA_graph);
    graph.nodes.push_back(nodeB_graph);

    graph.execute(&pool);

    // Wait for graph tasks to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    float sum = 0;
    for (size_t i=0; i<nodeB_graph->tensor.size(); i++) sum += nodeB_graph->tensor[i];
    std::cout << "Graph output sum: " << sum << std::endl;

    std::cout << "Press Enter to exit...\n";
    std::cin.get();
    return 0;
}

