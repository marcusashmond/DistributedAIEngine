#ifndef GRAPH_H
#define GRAPH_H

#include "GraphNode.h"
#include "ThreadPool.h"
#include <vector>
#include <memory>

class Graph {
public:
    std::vector<std::shared_ptr<GraphNode>> nodes;

    void execute(ThreadPool* pool) {
        for (auto& node : nodes) {
            node->compute(pool);
        }
    }
};

#endif
