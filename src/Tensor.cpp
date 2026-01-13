#include "Tensor.h"
#include <sstream>

Tensor::Tensor() {}

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
    size_t totalSize = 1;
    for (size_t dim : shape) {
        totalSize *= dim;
    }
    data.resize(totalSize, 0.0f);
}

float& Tensor::operator[](size_t index) {
    return data[index];
}

const float& Tensor::operator[](size_t index) const {
    return data[index];
}

const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::size() const {
    return data.size();
}

// Compact binary format:
//  - 4 bytes magic: 'TENS'
//  - 1 byte version (1)
//  - 1 byte dtype (1 = float32)
//  - 2 bytes reserved
//  - uint64_t dims (little-endian)
//  - dims * uint64_t shape entries (little-endian)
//  - uint64_t nelems (little-endian)
//  - raw float bytes (little-endian float32)
std::vector<char> Tensor::serializeBinary() const {
    std::vector<char> out;

    // header
    out.push_back('T'); out.push_back('E'); out.push_back('N'); out.push_back('S');
    out.push_back(1); // version
    out.push_back(1); // dtype = float32
    out.push_back(0); out.push_back(0); // reserved

    auto append_u64_le = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            out.push_back(static_cast<char>(v & 0xFF));
            v >>= 8;
        }
    };

    append_u64_le(static_cast<uint64_t>(shape.size()));
    for (size_t d : shape) append_u64_le(static_cast<uint64_t>(d));
    append_u64_le(static_cast<uint64_t>(data.size()));

    if (!data.empty()) {
        const char* bytes = reinterpret_cast<const char*>(data.data());
        out.insert(out.end(), bytes, bytes + data.size() * sizeof(float));
    }

    return out;
}

Tensor Tensor::deserializeBinary(const std::vector<char>& bytes) {
    size_t offset = 0;
    auto need = [&](size_t n) { return offset + n <= bytes.size(); };

    if (!need(8)) throw std::runtime_error("Invalid serialized tensor (header)");
    if (bytes[0] != 'T' || bytes[1] != 'E' || bytes[2] != 'N' || bytes[3] != 'S')
        throw std::runtime_error("Invalid tensor magic");
    uint8_t version = static_cast<uint8_t>(bytes[4]);
    if (version != 1) throw std::runtime_error("Unsupported tensor version");
    uint8_t dtype = static_cast<uint8_t>(bytes[5]);
    if (dtype != 1) throw std::runtime_error("Unsupported tensor dtype");
    offset = 8;

    auto read_u64_le = [&](uint64_t &out) -> bool {
        if (!need(8)) return false;
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) {
            v = (v << 8) | static_cast<uint8_t>(bytes[offset + i]);
        }
        out = v;
        offset += 8;
        return true;
    };

    uint64_t dims = 0;
    if (!read_u64_le(dims)) throw std::runtime_error("Invalid serialized tensor (dims)");

    std::vector<size_t> shape;
    for (uint64_t i = 0; i < dims; ++i) {
        uint64_t v = 0;
        if (!read_u64_le(v)) throw std::runtime_error("Invalid serialized tensor (shape)");
        shape.push_back(static_cast<size_t>(v));
    }

    uint64_t nelems = 0;
    if (!read_u64_le(nelems)) throw std::runtime_error("Invalid serialized tensor (nelems)");

    Tensor t(shape);
    size_t expected_bytes = static_cast<size_t>(nelems) * sizeof(float);
    if (!need(expected_bytes)) throw std::runtime_error("Invalid serialized tensor (data)");

    if (nelems > 0) {
        std::memcpy(t.data.data(), bytes.data() + offset, expected_bytes);
        offset += expected_bytes;
    }

    return t;
}

#include <sstream>

std::string Tensor::serialize() const {
    std::ostringstream out;

    // Write shape
    out << shape.size() << " ";
    for (size_t dim : shape) {
        out << dim << " ";
    }

    // Write data
    out << data.size() << " ";
    for (float v : data) {
        out << v << " ";
    }

    return out.str();
}

Tensor Tensor::deserialize(const std::string& buffer) {
    std::istringstream in(buffer);

    size_t shapeDims;
    in >> shapeDims;

    std::vector<size_t> shape(shapeDims);
    for (size_t i = 0; i < shapeDims; i++) {
        in >> shape[i];
    }

    Tensor tensor(shape);

    size_t dataSize;
    in >> dataSize;

    for (size_t i = 0; i < dataSize; i++) {
        in >> tensor.data[i];
    }

    return tensor;
}
