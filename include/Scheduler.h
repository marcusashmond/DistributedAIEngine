#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "ThreadPool.h"
#include "Task.h"

class Scheduler {
public:
    Scheduler(size_t numThreads);
    ~Scheduler();

    void submitTask(const Task& task);

private:
    ThreadPool threadPool;
};

#endif
