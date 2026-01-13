CXX = g++
CXXFLAGS = -std=c++17 -pthread -Iinclude -Wall -Wextra -O2
SRCS = $(wildcard src/*.cpp)
TARGET = DistributedAIEngine
BUILD_DIR = build
OUT = $(BUILD_DIR)/$(TARGET)

all: $(OUT)

$(OUT): $(SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(OUT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(OUT)

.PHONY: all clean
