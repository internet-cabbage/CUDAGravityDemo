# makefile made with assistance of this github exercise
# https://github.com/NCI900-Training-Organisation/intro-to-cuda/blob/main/exercises/exercise_13/Makefile


NVCC = nvcc
NVCCFLAGS = -I/home/linuxbrew/.linuxbrew/opt/cglm/include
SRCFILES = basicGravity.cu inputOutput.c tree.cu initialConditions.c sorting.cu
TARGET = gravitySim

STD = -std=c++17
ARCH = -arch=sm_86 # The architecture being compiled to

# Compile types

WARNINGS = 	-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wshadow \
			-Xcompiler=-Wpointer-arith -Xcompiler=-Wunreachable-code \
			-Xcompiler=-Wno-unused-parameter \

RELEASE = -O3 -Xcompiler=-ffast-math -Xcompiler=-march=native

QUICK = -O0

DEBUG = -g -lineinfo -Xcompiler=-Og -Xcompiler=-fsanitize=undefined -Xcompiler=-fsanitize=address -Xcompiler=-fno-sanitize-recover=all

default: $(SRCFILES)
	$(NVCC) $(NVCCFLAGS) $(STD) $(ARCH) $(RELEASE) $(WARNINGS) $^ -o $(TARGET)

quick: $(SRCFILES)
	$(NVCC) $(NVCCFLAGS) $(STD) $(ARCH) $(QUICK) $(WARNINGS) $^ -o $(TARGET)
debug: $(SRCFILES)
	$(NVCC) $(NVCCFLAGS) $(STD) $(ARCH) $(DEBUG) $(WARNINGS) $^ -o $(TARGET)
run:
	./$(TARGET)
computeDebug:
	compute-sanitizer --tool racecheck ./gravitySim
clean: 
	rm -f $(TARGET)
push:
	rsync -av --partial --progress outputDump.bin luthaisb@100.80.190.16:~/Code/C++/OpenGL/BasicLighting/data

memCheck: $(SRCFILES)
	$(NVCC) $(NVCCFLAGS) -Xptxas -v $(STD) $(ARCH) $(DEBUG) $(WARNINGS) $^ -o $(TARGET)