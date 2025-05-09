# Application
CXX := g++
NVCC := /usr/local/cuda/bin/nvcc
MOC = moc
INCLUDES := \
	-Iinclude \
	-I/usr/local/cuda-11.8/targets/x86_64-linux/include
LIB_DIRS := \
	-L/usr/lib/x86_64-linux-gnu \
	-L/usr/local/cuda/lib64
CXXFLAGS := -std=c++17 -Wall -Wextra -fPIC $(shell pkg-config --cflags Qt5Core Qt5Gui Qt5Widgets) $(INCLUDES) $(LIB_DIRS)
NVCCFLAGS := -Xcompiler -fPIC -lineinfo -Wno-deprecated-declarations $(INCLUDES)
LDFLAGS := $(shell pkg-config --libs Qt5Core Qt5Gui Qt5Widgets) -lcufft -lcudart

SRC_DIR := src
MOC_DIR := moc
INCLUDE_DIR := include

TARGET := spectrum

CU_SOURCES := $(SRC_DIR)/fft_magnitude.cu
CPP_SOURCES := $(SRC_DIR)/spectrum.cpp $(SRC_DIR)/sineanimation.cpp $(MOC_DIR)/sineanimation.moc.cpp
CU_OBJECTS := $(CU_SOURCES:.cu=.o)
CPP_OBJECTS := $(CPP_SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MOC_DIR)/%.moc.cpp: $(INCLUDE_DIR)/%.h
	-mkdir -p moc
	$(MOC) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	-rm -f $(OBJECTS) $(TARGET)
