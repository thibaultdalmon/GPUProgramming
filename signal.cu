#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define PI 3.14159265

//#define GPU_COMPUTING

__global__ void conv(float *tab, int N, float *filter, int s, float *output);
void box_filter(float *filter, int size);
void gaussian_filter(float *filter, int size);
void conv_GPU(float *tab, int N, float *filter, int s, float *tab_filtered, int N_threads);
void conv_CPU(float *tab, int N, float *filter, int s, float *tab_filtered);

int main(int argc, char const *argv[]) {
  FILE * fp;
  int32_t N = (int32_t) atoi(argv[1]);
  int32_t N_threads = (int32_t)atoi(argv[2]);
  int32_t s = (int32_t) atoi(argv[3]);
  float freq1 = atof(argv[4]);
  float freq2 = atof(argv[5]);

  float phi = 1.0;
  float *tab_CPU = (float *) malloc(N*sizeof(float));
  float *tab_CPU_box = (float *) malloc(N*sizeof(float));
  float *tab_CPU_gaussian = (float *) malloc(N*sizeof(float));
  float filter[2*s+1];

  for (int i=0; i<N; i++){
    tab_CPU[i] = sin(2*PI*i/freq1) + sin(2*PI*i/freq2+phi);
  }

  for (int i=1; i<31; i++){
    s = (int32_t) 20*i;
    printf("Iteration %d, ", i);

    gaussian_filter(filter, s);
    printf("GPU_gaussian: ");
    conv_GPU(tab_CPU, N, filter, s, tab_CPU_gaussian, N_threads);
    printf(", CPU_gaussian: ");
    conv_CPU(tab_CPU, N, filter, s, tab_CPU_gaussian);

    box_filter(filter, s);
    printf(", GPU_box: ");
    conv_GPU(tab_CPU, N, filter, s, tab_CPU_box, N_threads);
    printf(", CPU_box: ");
    conv_CPU(tab_CPU, N, filter, s, tab_CPU_box);

    printf("\n");
  }

  // for (int i=0; i<2*s+1; i++){
  //   printf("%f ", filter[i]);
  // }

  fp = fopen ("signal.data", "w+");
  for (int i=0; i<200; i++){
    fprintf(fp, "%f %f %f\n", tab_CPU[i], tab_CPU_box[i], tab_CPU_gaussian[i]);
  }
  fclose(fp);

  return 0;
}


__global__ void conv(float *tab, int N, float *filter, int s, float *output){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx<=N){
    output[idx] = filter[s]*tab[idx];
    for (int i=1; i<s+1; i++){
      if (idx-i >= 0) output[idx] += filter[s+i]*tab[idx-i];
      else output[idx] += filter[s+i]*tab[idx-i+N];
      if(idx+i < N) output[idx] += filter[s-i]*tab[idx+i];
      else output[idx] += filter[s+i]*tab[idx+i-N];
    }
  }
}

void box_filter(float *filter, int size) {
  for (int i=0; i<2*size+1; i++){
    filter[i] = 1/(float)(2*size+1);
  }
}

void gaussian_filter(float *filter, int size) {
  float s = (float) (2*size+1);
  float sum = 0;
  for (int i=0; i<2*size+1; i++){
    filter[i] =  exp(-(i-size)*(i-size) / (2*s*s));
    sum += filter[i];
  }
  for (int i=0; i<2*size+1; i++){
    filter[i] /= sum;
  }
}

void conv_GPU(float *tab, int N, float *filter, int s, float *tab_filtered, int N_threads) {
  clock_t start, finish;
  double duration;
  start = clock();

  float *tab_GPU;
  float *output_GPU;
  float *filter_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * sizeof(float));
  cudaMalloc(&output_GPU, N * sizeof(float));
  cudaMalloc(&filter_GPU, (2*s+1) * sizeof(float));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_GPU, tab, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_GPU, filter,  (2*s+1) * sizeof(float), cudaMemcpyHostToDevice);

  int threadsPerBlock = N_threads;
  int blocksPerGrid =
            (int) ceil(N / (float)threadsPerBlock);

  conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, filter_GPU, s, output_GPU);

  cudaMemcpy(tab_filtered, output_GPU, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);
  cudaFree(filter_GPU);
  cudaFree(output_GPU);

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f",duration);
}

void conv_CPU(float *tab, int N, float *filter, int s, float *tab_filtered){
  clock_t start, finish;
  double duration;
  start = clock();

  double t = (double) time(NULL);
  for (int idx=0; idx<N; idx++){
    tab_filtered[idx] = filter[s]*tab[idx];
    for (int i=1; i<s+1; i++){
      tab_filtered[idx] += filter[s+i]*tab[(idx-i)%N];
      tab_filtered[idx] += filter[s-i]*tab[(idx+i)%N];
    }
  }

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f",duration);
}
