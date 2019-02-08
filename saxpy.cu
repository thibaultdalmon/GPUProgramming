#include<stdio.h>
#include<stdint.h>

__global__ void saxpy(int32_t *tab, int32_t N, int32_t a, int32_t b);

int main(int argc, char const *argv[]) {
  int32_t N = (int32_t) atoi(argv[1]);
  int32_t a = (int32_t) atoi(argv[2]);
  int32_t b = (int32_t) atoi(argv[3]);
  int32_t N_threads = (int32_t) atoi(argv[4]);

  int32_t tab_CPU[N];
  for (int i=0; i<N; i++){
    tab_CPU[i] = (int32_t) i;
  }

  int32_t *tab_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(int32_t));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab_CPU, N * sizeof(int32_t), cudaMemcpyHostToDevice);

  int32_t threadsPerBlock = N_threads;
  int32_t blocksPerGrid =
            (int32_t) ceil(N / (float)threadsPerBlock);

  saxpy<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, a, b);

  cudaMemcpy(tab_CPU, tab_GPU, N * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);

  for (int i=0; i<20; i++){
    printf("%d ",tab_CPU[i]);
  }

  return 0;
}


__global__ void saxpy(int32_t *tab, int32_t N, int32_t a, int32_t b){
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N)
    tab[idx] = a * tab[idx] + b;
}
