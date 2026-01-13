#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "ThreadPool.h"

int main() {
    ThreadPool pool(4);
    const int total = 8;
    std::atomic<int> completed{0};
    std::mutex m;
    std::condition_variable cv;

    for (int i = 0; i < total; ++i) {
        pool.enqueue([i, &completed, &cv] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            int v = ++completed;
            if (v == 8) cv.notify_one();
            std::cout << "Task " << i << " done\n";
        });
    }

    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return completed.load() >= total; });

    std::cout << "All tasks completed\n";
    return 0;
}
