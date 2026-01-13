#include "Node.h"
#include "Tensor.h"
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

    // Set up two nodes on different ports: nodeA will send tensor to nodeB
    Node nodeA(5001, 4, 1);
    Node nodeB(5002, 4, 2);
    nodeA.startServer();
    nodeB.startServer();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Demonstrate broadcasting a tensor from nodeA to two receiving clients
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Start two receiver threads that connect to nodeA and wait for a tensor
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

        // Read size_t length (sender uses size_t)
        size_t size = 0;
        ssize_t r = ::recv(sock, &size, sizeof(size_t), MSG_WAITALL);
        if (r <= 0) { std::cerr << "recv size failed\n"; close(sock); return; }

        std::vector<char> buffer(size);
        r = ::recv(sock, buffer.data(), size, MSG_WAITALL);
        if (r <= 0) { std::cerr << "recv payload failed\n"; close(sock); return; }

        Tensor t = Tensor::deserializeBinary(buffer);
        std::cout << "Receiver " << idx << " got tensor of size " << t.size() << ": ";
        for (size_t i = 0; i < t.size(); ++i) std::cout << t[i] << " ";
        std::cout << std::endl;

        close(sock);
    };

    std::thread r1(receiver, 1);
    std::thread r2(receiver, 2);

    Tensor t({2, 3});
    for (size_t i = 0; i < t.size(); i++) {
        t[i] = static_cast<float>(i);
    }

    nodeA.broadcastTensor(t);

    r1.join();
    r2.join();

    std::cout << "Press Enter to exit...\n";
    std::cin.get();
    return 0;
}

