export WORKSPACE=$(shell pwd)
export USER_NAME=$(shell whoami)
export USER_UID=$(shell id -u)
export USER_GID=$(shell id -g)

PROJECT_NAME := gpu
DEV_SERVICE := dev
DOCKER_COMPOSE_FILE := docker/gpu.docker-compose.yml
DOCKER_COMPOSE_CMD := docker compose --project-name $(PROJECT_NAME) --file $(DOCKER_COMPOSE_FILE)

build:
	$(DOCKER_COMPOSE_CMD) \
		--progress plain \
		build

shell:
	@cid="$$( $(DOCKER_COMPOSE_CMD) ps -q $(DEV_SERVICE) )"; \
	if [ -n "$$cid" ]; then \
		echo "Container '$(DEV_SERVICE)' is already running."; \
		$(DOCKER_COMPOSE_CMD) exec $(DEV_SERVICE) /bin/bash; \
	else \
		$(DOCKER_COMPOSE_CMD) up -d $(DEV_SERVICE) && $(DOCKER_COMPOSE_CMD) exec $(DEV_SERVICE) /bin/bash; \
	fi

stop:
	$(DOCKER_COMPOSE_CMD) down


# Application
CXX = g++
MOC = moc
INCLUDES := \
	-Iinclude
LIB_DIRS := \
	-L/usr/lib/x86_64-linux-gnu
CXXFLAGS = -std=c++17 -Wall -Wextra -fPIC $(shell pkg-config --cflags Qt5Core Qt5Gui Qt5Widgets) $(INCLUDES) $(LIB_DIRS)
LDFLAGS = $(shell pkg-config --libs Qt5Core Qt5Gui Qt5Widgets)

SRC_DIR := src
MOC_DIR := moc
INCLUDE_DIR := include

TARGET = spectrum

SOURCES = $(SRC_DIR)/main.cpp $(SRC_DIR)/sineanimation.cpp $(MOC_DIR)/sineanimation.moc.cpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MOC_DIR)/%.moc.cpp: $(INCLUDE_DIR)/%.h
	$(MOC) $< -o $@

clean:
	-rm -f $(OBJECTS) $(TARGET)
