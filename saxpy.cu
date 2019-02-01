#include<cstdio.h>

int main(int argc, char const *argv[]) {
  int N = argv[1];
  int a = argv[2];
  int b = argv[3];
  int tab_CPU[N];

  for (int i=0; i<N; i++){
    tab_CPU[i] = i;
  }

  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(int));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab_CPU, N * sizeof(int), cudaMemcpyHostToDevice);

  saxpy<<<1,N>>>(tab_GPU, N, a, b);

  cudaMemcpy(tab_CPU, tab_GPU, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);

  return 0;
}


__global__ void saxpy(int *tab, int N, int a, int b){
  int idx = threadIdx.x;
  tab[idx+1] = a * tab[idx+1] + b;
}
