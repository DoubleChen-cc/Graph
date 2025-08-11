DEBUG =

OPTIONS = -Xptxas -v
GPU_MATCH = src/gpu_match.cu

define compile_cu_test
        nvcc -std=c++17 $(DEBUG) $(OPTIONS) -arch=compute_70 $(1) cu_test.cu -o $(2)
endef

define compile_gpu_match
        nvcc -std=c++17 $(DEBUG) $(OPTIONS) -arch=compute_70 -dc -I. $(1) -o $(2)
endef

.PHONY:all
all:bin/reint
bin/reint.exe:bin/reint.o
        $(call compile_cu_test,$<,$@)
bin/reint.o:
        $(call compile_gpu_match,src/gpu_match.cu,$@)

.PHONY:clean
clean:
    rm -f bin/reint.o bin/reint.exe