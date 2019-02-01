#include<stdio.h>
#include<math.h>

#define PI 3.14159265

__global__ void signal(int *tab, int N, int a, int b);

int main(int argc, char const *argv[]) {
  int N = atoi(argv[1]);
  int N_threads = atoi(argv[2]);
  double freq1 = atof(argv[3]);
  double freq2 = atof(argv[4]);


  double tab_CPU[N];
  for (int i=0; i<N; i++){
    tab_CPU[i] = sin(i*2*PI*freq1)+ sin(i*2*PI*freq2);
  }

  double *tab_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(double));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab_CPU, N * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock = N_threads;
  int blocksPerGrid =
            (int) ceil(N / (float)threadsPerBlock);

  saxpy<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, a, b);

  cudaMemcpy(tab_CPU, tab_GPU, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);

  for (int i=0; i<20; i++){
    printf("%d ",tab_CPU[i]);
  }

  return 0;
}


__global__ void signal(int *tab, int N, int a, int b){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N)
    tab[idx] = a * tab[idx] + b;
}
