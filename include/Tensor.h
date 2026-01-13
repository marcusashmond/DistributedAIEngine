#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <cstdint>

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<size_t>& shape);

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    const std::vector<size_t>& getShape() const;
    size_t size() const;

    // Serialize tensor to bytes (shape followed by raw float data)
    std::vector<char> serializeBinary() const;

    // Reconstruct tensor from bytes produced by serializeBinary()
    static Tensor deserializeBinary(const std::vector<char>& bytes);

    // Text-based serialization helpers (human-readable)
    std::string serialize() const;
    static Tensor deserialize(const std::string& buffer);

private:
    std::vector<size_t> shape;
    std::vector<float> data;
};

#endif
