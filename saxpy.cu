#include<stdio.h>

__global__ void saxpy(int *tab, int N, int a, int b);

int main(int argc, char const *argv[]) {
  int N = 32;
  int a = 2;
  int b = 1;
  int tab_CPU[N];

  for (int i=0; i<N; i++){
    tab_CPU[i] = i;
  }

  int *tab_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(int));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab_CPU, N * sizeof(int), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid =
            ceil(N / threadsPerBlock);

  saxpy<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, a, b);

  cudaMemcpy(tab_CPU, tab_GPU, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);

  for (int i=0; i<20; i++){
    printf("%d ",tab_CPU[i]);
  }

  return 0;
}


__global__ void saxpy(int *tab, int N, int a, int b){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N)
    tab[idx] = a * tab[idx] + b;
}
