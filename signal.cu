#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265

__global__ void conv(double *tab, int N, double *filter, int s, double *output);
void box_filter(double *filter, int size);
void gaussian_filter(double *filter, int size);
void conv_GPU(double *tab, int N, double *filter, int s, double *tab_filtered, int N_threads);

int main(int argc, char const *argv[]) {
  FILE * fp;
  int N = atoi(argv[1]);
  int N_threads = atoi(argv[2]);
  int s = atoi(argv[3]);
  double freq1 = atof(argv[4]);
  double freq2 = atof(argv[5]);

  double phi = 1.0;
  double tab_CPU[N];
  double tab_CPU_box[N];
  double tab_CPU_gaussian[N];
  double filter[2*s+1];

  for (int i=0; i<N; i++){
    tab_CPU[i] = sin(2*PI*i/freq1) + sin(2*PI*i/freq2+phi);
    tab_CPU_box[i] = tab_CPU[i];
    tab_CPU_gaussian[i] = tab_CPU[i];
  }
  gaussian_filter(filter, s);
  // for (int i=0; i<2*s+1; i++){
  //   printf("%lf ", filter[i]);
  // }

  conv_GPU(tab_CPU, N, filter, s, tab_CPU_gaussian, N_threads);

  box_filter(filter, s);
  conv_GPU(tab_CPU, N, filter, s, tab_CPU_box, N_threads);

  fp = fopen ("signal.data", "w+");
  for (int i=0; i<200; i++){
    fprintf(fp, "%lf %lf %lf\n", tab_CPU[i], tab_CPU_box[i], tab_CPU_gaussian[i]);
  }
  fclose(fp);

  return 0;
}


__global__ void conv(double *tab, int N, double *filter, int s, double *output){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N)
    output[idx] = filter[s]*tab[idx];
    for (int i=1; i<s+1; i++){
      if (idx-i>=0) output[idx] += filter[s+i]*tab[idx-i];
      if (idx+i<N) output[idx] += filter[s-i]*tab[idx+i];
    }
}

void box_filter(double *filter, int size) {
  for (int i=0; i<2*size+1; i++){
    filter[i] = 1/(float)(2*size+1);
  }
}

void gaussian_filter(double *filter, int size) {
  double s = (double) (2*size+1);
  double sum = 0;
  for (int i=0; i<2*size+1; i++){
    filter[i] =  exp(-(i-size)*(i-size) / (2*s*s));
    sum += filter[i];
  }
  for (int i=0; i<2*size+1; i++){
    filter[i] /= sum;
  }
}

void conv_GPU(double *tab, int N, double *filter, int s, double *tab_filtered, int N_threads) {
  double *tab_GPU;
  double *output_GPU;
  double *filter_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(double));
  cudaMalloc(&output_GPU, N * sizeof(double));
  cudaMalloc(&filter_GPU, (2*s+1) * sizeof(double));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(output_GPU, tab, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_GPU, filter,  (2*s+1) * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock = N_threads;
  int blocksPerGrid =
            (int) ceil(N / (float)threadsPerBlock);

  conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, filter_GPU, s, output_GPU);

  cudaMemcpy(tab_filtered, output_GPU, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);
  cudaFree(filter_GPU);
  cudaFree(output_GPU);
}
