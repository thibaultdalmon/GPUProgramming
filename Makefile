all: saxpy signal image

saxpy: saxpy.cu
	nvcc -lm -o saxpy saxpy.cu

signal: signal.cu
	nvcc -lm -o signal signal.cu

image: image.cu
	nvcc -lm -o image image.cu

