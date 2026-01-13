#include "Scheduler.h"

Scheduler::Scheduler(size_t numThreads) : threadPool(numThreads) {}

Scheduler::~Scheduler() {
    // ThreadPool destructor handles stopping threads
}

void Scheduler::submitTask(const Task& task) {
    threadPool.enqueue([task]() {
        task.work(task.tensor);
    });
}
