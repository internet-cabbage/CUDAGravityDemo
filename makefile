# makefile made with assistance of this github exercise
# https://github.com/NCI900-Training-Organisation/intro-to-cuda/blob/main/exercises/exercise_13/Makefile


NVCC = nvcc
SRCFILES = basicGravity.cu inputOutput.c
TARGET = gravitySim

STD = -std=c++17
ARCH = -arch=sm_86 # The architecture being compiled to

# Compile types

WARNINGS = 	-Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wshadow \
			-Xcompiler=-Wpointer-arith -Xcompiler=-Wunreachable-code \

RELEASE = -O3 -Xcompiler=-ffast-math -Xcompiler=-march=native

DEBUG = -g -lineinfo -Xcompiler=-Og -Xcompiler=-fsanitize=undefined -Xcompiler=-fno-sanitize-recover=all

default: $(SRCFILES)
	$(NVCC) $(STD) $(ARCH) $(RELEASE) $(WARNINGS) $^ -o $(TARGET)

debug: $(SRCFILES)
	$(NVCC) $(STD) $(ARCH) $(DEBUG) $(WARNINGS) $^ -o $(TARGET)
run:
	./$(TARGET)
computeSanitize:
	compute-sanitizer ./gravitySim
clean: 
	rm -f $(TARGET)
push:
	rsync -av --partial --progress outputDump.bin luthaisb@100.80.190.16:~/Code/C++/OpenGL/BasicLighting/data

memCheck:
	$(NVCC) -Xptxas -v $(SRCFILES)