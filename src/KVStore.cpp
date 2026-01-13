#include "KVStore.h"
#include <fstream>
#include <iostream>

void KVStore::put(const std::string& key, const Tensor& tensor) {
    std::lock_guard<std::mutex> lock(storeMutex);
    store[key] = tensor;
}

bool KVStore::get(const std::string& key, Tensor& outTensor) {
    std::lock_guard<std::mutex> lock(storeMutex);

    auto it = store.find(key);
    if (it == store.end()) {
        return false;
    }

    outTensor = it->second;
    return true;
}

bool KVStore::saveToDisk(const std::string& key) {
    std::lock_guard<std::mutex> lock(storeMutex);

    auto it = store.find(key);
    if (it == store.end()) return false;

    std::string filename = "checkpoints/" + key + ".chk";
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    auto buffer = it->second.serializeBinary();
    out.write(buffer.data(), buffer.size());
    return true;
}

bool KVStore::loadFromDisk(const std::string& key) {
    std::lock_guard<std::mutex> lock(storeMutex);

    std::string filename = "checkpoints/" + key + ".chk";
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    std::vector<char> buffer(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>()
    );

    Tensor tensor = Tensor::deserializeBinary(buffer);
    store[key] = tensor;
    std::cout << "Checkpoint loaded: " << key << std::endl;
    return true;
}
