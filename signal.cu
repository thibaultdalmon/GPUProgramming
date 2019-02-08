#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265

__global__ void conv(double *tab, int N, double *filter, int s);
void box_filter(double *filter, int size);

int main(int argc, char const *argv[]) {
  FILE * fp;
  int N = atoi(argv[1]);
  int N_threads = atoi(argv[2]);
  int s = atoi(argv[3]);
  double freq1 = atof(argv[4]);
  double freq2 = atof(argv[5]);

  double phi = 1.0;
  double tab_CPU_filtered[N];
  double tab_CPU[N];
  double filter[2*s+1];

  for (int i=0; i<N; i++){
    tab_CPU[i] = sin(2*PI*i/freq1)+ sin(2*PI*i/freq2+phi);
    tab_CPU_filtered[i] = tab_CPU[i];
  }
  box_filter(filter, s);

  double *tab_GPU;
  double *filter_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(double));
  cudaMalloc(&filter_GPU, (2*s+1) * sizeof(double));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab_CPU, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_GPU, filter,  (2*s+1) * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock = N_threads;
  int blocksPerGrid =
            (int) ceil(N / (float)threadsPerBlock);

  conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, filter_GPU, s);

  cudaMemcpy(tab_CPU_filtered, tab_GPU, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);
  cudaFree(filter_GPU);

  fp = fopen ("signal.data", "w+");
  for (int i=0; i<200; i++){
    fprintf(fp, "%lf %lf\n", tab_CPU[i], tab_CPU_filtered[i]);
  }
  fclose(fp);

  return 0;
}


__global__ void conv(double *tab, int N, double *filter, int s){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N)
    tab[idx] = filter[s]*tab[idx];
    for (int i=1; i<s+1; i++){
      if (idx-s>=0) tab[idx] += filter[s+i]*tab[idx-i];
      if (idx+s<N) tab[idx] += filter[s-i]*tab[idx+i];
    }
}

void box_filter(double *filter, int size) {
  for (int i=0; i<2*size+1; i++){
    filter[i] = 1/(float)(2*size+1);
  }
}
