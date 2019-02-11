#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

__global__ void conv(float *tab, int N, float *filter, int s, float *output);
void box_filter(float *filter, int size);
void gaussian_filter(float *filter, int size);
void conv_GPU(float *tab, int N, float *filter, int s, float *tab_filtered, int N_threads);
struct Image {
  int *img;
  int height;
  int width;
  int depth;
};
void open_img(Image * image);
void write_img(Image * image);


int main(int argc, char const *argv[]) {
  Image image;
  Image * image_addr = &image;

  open_img(image_addr);
  write_img(image_addr);

  return 0;
}

void open_img(Image *image){
  FILE *fp;
  int height, width, depth;
  int c;

  fp = fopen("image_256x256.pgm", "r");
  fscanf(fp, "P%d\n", &c);
  fscanf(fp, "%d %d\n", &height, &width);
  fscanf(fp, "%d\n", &depth);

  image->img = (int *) malloc(height*width*sizeof(int*));
  image->height = height;
  image->width = width;
  image->depth = depth;

  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      fscanf(fp, "%d\n", &c);
      *(image->img+i*width+j) = c;
    }
  }
  fclose(fp);
}

void write_img(Image *image){
  FILE *fp;
  int height = image->height;
  int width = image->width;
  int depth = image->depth;

  fp = fopen ("test.pgm", "w+");
  fprintf(fp, "P2\n%d %d\n%d\n", height, width, depth);

  for (int i=0; i<height; i++){
    for (int j=0; j<width; j++)
      fprintf(fp, "%d\n", *(image->img+i*width+j));
  }
  fclose(fp);
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
