#ifndef GRAPHNODE_H
#define GRAPHNODE_H

#include "Tensor.h"
#include "ThreadPool.h"
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <iostream>
#include <thread>

class GraphNode {
public:
    std::string name;
    Tensor tensor;
    std::vector<std::shared_ptr<GraphNode>> inputs;
    std::function<void()> operation; // define how to compute tensor

    void compute(ThreadPool* pool) {
        if (!operation) return;

        pool->enqueue([this]() {
            operation();
            std::cout << "Node " << name << " computed on thread "
                      << std::this_thread::get_id() << std::endl;
        });
    }
};

#endif
