#ifndef TASK_H
#define TASK_H

#include <functional>
#include <string>
#include "Tensor.h"

enum class TaskType {
    COMPUTE,
    IO
};

struct Task {
    TaskType type;
    std::string name;
    Tensor tensor;
    std::function<void(const Tensor&)> work;
};

#endif
