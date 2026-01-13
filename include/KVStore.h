#ifndef KVSTORE_H
#define KVSTORE_H

#include <unordered_map>
#include <string>
#include <mutex>
#include "Tensor.h"

class KVStore {
public:
    void put(const std::string& key, const Tensor& tensor);
    bool get(const std::string& key, Tensor& outTensor);
    bool saveToDisk(const std::string& key);
    bool loadFromDisk(const std::string& key);

private:
    std::unordered_map<std::string, Tensor> store;
    std::mutex storeMutex;
};

#endif
