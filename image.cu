#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

__global__ void conv(float *tab, int N, int M,
  float *filter, int s, float *output);
__global__ void conv_bil(float *tab, int N, int M,
  float *filter, int s, float *output, float r);
void box_filter(float *filter, int size);
void gaussian_filter(float *filter, int size);
void conv_GPU(float *tab, int N, int M, float *filter,
  int s, float *tab_filtered, int N_threads);

typedef struct {
  float *img;
  int height;
  int width;
  int depth;
} Image;

void open_img(Image * image);
void write_img(Image * image, int file_idx);
void box_filtering(Image *image, int s);
void gaussian_filtering(Image *image, int s);
void bilateral_filtering(Image *image, int s, float r);


int main(int argc, char const *argv[]) {
  Image image;
  Image * image_addr = &image;
  int s = 5;

  open_img(image_addr);
  box_filtering(image_addr, s);
  write_img(image_addr, 0);

  open_img(image_addr);
  gaussian_filtering(image_addr, s);
  write_img(image_addr, 1);

  open_img(image_addr);
  bilateral_filtering(image_addr, s, 10);
  write_img(image_addr, 2);

  return 0;
}

void box_filtering(Image *image, int s){
  float * filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
  float * img_filtered = (float*) malloc(image->height*image->width
    *sizeof(float));
  box_filter(filter, s);
  conv_GPU(image->img, image->height, image->width,
    filter, s, img_filtered, 16);
  printf("\n");
  image->img = img_filtered;
}

void gaussian_filtering(Image *image, int s){
  float * filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
  float * img_filtered = (float*) malloc(image->height*image->width
    *sizeof(float));
  gaussian_filter(filter, s);
  conv_GPU(image->img, image->height, image->width,
    filter, s, img_filtered, 16);
  printf("\n");
  image->img = img_filtered;
}

void bilateral_filtering(Image *image, int s, float r){
  clock_t start, finish;
  double duration;
  start = clock();

  float * filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
  float * img_filtered = (float*) malloc(image->height*image->width
    *sizeof(float));
  int N = image->height;
  int M = image->width;
  int N_threads = 16;
  float size = (float) (2*s+1);

  for (int i=0; i<2*s+1; i++){
    for (int j=0; j<2*s+1; j++){
      filter[i*(2*s+1)+j] =  exp(-((i-s)*(i-s)+(j-s)*(j-s))/ (2*size*size));
    }
  }

  float *tab_GPU;
  float *output_GPU;
  float *filter_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * M * sizeof(float));
  cudaMalloc(&output_GPU, N * M * sizeof(float));
  cudaMalloc(&filter_GPU, (2*s+1) * (2*s+1) * sizeof(float));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, image->img, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_GPU, img_filtered, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_GPU, filter,  (2*s+1) * (2*s+1) *
    sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(N_threads, N_threads, 1);
  dim3 blocksPerGrid((int) ceil(N / (float)threadsPerBlock.x),
            (int) ceil(M / (float)threadsPerBlock.y), 1);

  conv_bil<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, M, filter_GPU, s, output_GPU, r);
  if ( cudaSuccess != cudaGetLastError() )
    printf( "Error!\n" );

  cudaMemcpy(img_filtered, output_GPU, N * M * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);
  cudaFree(filter_GPU);
  cudaFree(output_GPU);

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f\n",duration);
  image->img = img_filtered;
}

void open_img(Image *image){
  FILE *fp;
  int height, width, depth;
  int c;

  fp = fopen("image_256x256.pgm", "r");
  fscanf(fp, "P%d\n", &c);
  fscanf(fp, "%d %d\n", &height, &width);
  fscanf(fp, "%d\n", &depth);

  image->img = (float *) malloc(height*width*sizeof(float*));
  image->height = height;
  image->width = width;
  image->depth = depth;

  for(int i=0; i<height; i++){
    for(int j=0; j<width; j++){
      fscanf(fp, "%d\n", &c);
      *(image->img+i*width+j) = (float) c;
    }
  }
  fclose(fp);
}

void write_img(Image *image, int file_idx){
  FILE *fp;
  int height = image->height;
  int width = image->width;
  int depth = image->depth;

  switch (file_idx) {
    case 0:
      fp = fopen ("test_box.pgm", "w+");
      break;
    case 1:
      fp = fopen ("test_gauss.pgm", "w+");
      break;
    case 2:
      fp = fopen ("test_bil.pgm", "w+");
      break;
    default :
      fp = fopen ("test.pgm", "w+");
  }

  fprintf(fp, "P2\n%d %d\n%d\n", height, width, depth);

  for (int i=0; i<height; i++){
    for (int j=0; j<width; j++)
      fprintf(fp, "%d\n", (int) *(image->img+i*width+j));
  }
  fclose(fp);
}

__global__ void conv(float *tab, int N, int M,
  float *filter, int s, float *output){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx<=N){
    if (idy<=M){
      output[idx*M+idy] = filter[s]*tab[idx*M+idy];
      float sum = filter[s];
      for (int i=1; i<s+1; i++){
        int x_big = -N*(idx+i>=N);
        int x_small = N*(idx-i<0);
        float t = (filter[(s+i)*(2*s+1)+s]);
        float b = (filter[(s-i)*(2*s+1)+s]);

        output[idx*M+idy] += (t*tab[(idx+i+x_big)*M]);
        output[idx*M+idy] += (b*tab[(idx-i+x_small)*M]);

        for (int j=1; j<s+1; j++){
          int y_big = -M*(idy+j>=M);
          int y_small = M*(idy-j<0);
          float tl = (filter[(s+i)*(2*s+1)+(s+j)]);
          float tr = (filter[(s+i)*(2*s+1)+(s-j)]);
          float bl = (filter[(s-i)*(2*s+1)+(s+j)]);
          float br = (filter[(s-i)*(2*s+1)+(s-j)]);

          output[idx*M+idy] += (tl*tab[(idx-i+x_small)*M+idy-j+y_small]);
          output[idx*M+idy] += (tr*tab[(idx-i+x_small)*M+idy+j+y_big]);
          output[idx*M+idy] += (bl*tab[(idx+i+x_big)*M+idy-j+y_small]);
          output[idx*M+idy] += (br*tab[(idx+i+x_big)*M+idy+j+y_big]);

          if (i==1) {
            float l = (filter[(s)*(2*s+1)+(s+j)]);
            float r = (filter[(s)*(2*s+1)+(s-j)]);

            output[idx*M+idy] += (l*tab[(idx)*M+idy-j+y_small]);
            output[idx*M+idy] += (r*tab[(idx)*M+idy+j+y_big]);

            sum += l+r;
          }

          sum += tl+tr+bl+br;
        }
        sum += t+b;
      }
      output[idx*M+idy] /= sum;
    }
  }
}

__global__ void conv_bil(float *tab, int N, int M,
  float *filter, int s, float *output, float r){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  float var = 2*r*r;

  if (idx<=N){
    if (idy<=M){
      output[idx*M+idy] = filter[s]*tab[idx*M+idy];
      float sum = filter[s];
      for (int i=1; i<s+1; i++){
        int x_big = -N*(idx+i>=N);
        int x_small = N*(idx-i<0);
        float t = (filter[(s+i)*(2*s+1)+s]
          *exp(-(tab[(idx+i+x_big)*M]-tab[idx*M+idy])
          *(tab[(idx+i+x_big)*M]-tab[idx*M+idy])/var)
        );
        float b = (filter[(s-i)*(2*s+1)+s]
          *exp(-(tab[(idx-i+x_small)*M]-tab[idx*M+idy])
          *(tab[(idx-i+x_small)*M]-tab[idx*M+idy])/var)
        );

        output[idx*M+idy] += (t*tab[(idx+i+x_big)*M]);
        output[idx*M+idy] += (b*tab[(idx-i+x_small)*M]);

        for (int j=1; j<s+1; j++){
          int y_big = -M*(idy+j>=M);
          int y_small = M*(idy-j<0);
          float tl = (filter[(s+i)*(2*s+1)+(s+j)]
            *exp(-(tab[(idx-i+x_small)*M+idy-j+y_small]-tab[idx*M+idy])
            *(tab[(idx-i+x_small)*M+idy-j+y_small]-tab[idx*M+idy])/var)
          );
          float tr = (filter[(s+i)*(2*s+1)+(s-j)]
            *exp(-(tab[(idx-i+x_small)*M+idy+j+y_big]-tab[idx*M+idy])
            *(tab[(idx-i+x_small)*M+idy+j+y_big]-tab[idx*M+idy])/var)
          );
          float bl = (filter[(s-i)*(2*s+1)+(s+j)]
            *exp(-(tab[(idx+i+x_big)*M+idy-j+y_small]-tab[idx*M+idy])
            *(tab[(idx+i+x_big)*M+idy-j+y_small]-tab[idx*M+idy])/var)
          );
          float br = (filter[(s-i)*(2*s+1)+(s-j)]
            *exp(-(tab[(idx+i+x_big)*M+idy+j+y_big]-tab[idx*M+idy])
            *(tab[(idx+i+x_big)*M+idy+j+y_big]-tab[idx*M+idy])/var)
          );

          output[idx*M+idy] += (tl*tab[(idx-i+x_small)*M+idy-j+y_small]);
          output[idx*M+idy] += (tr*tab[(idx-i+x_small)*M+idy+j+y_big]);
          output[idx*M+idy] += (bl*tab[(idx+i+x_big)*M+idy-j+y_small]);
          output[idx*M+idy] += (br*tab[(idx+i+x_big)*M+idy+j+y_big]);

          if (i==1) {
            float l = (filter[(s)*(2*s+1)+(s+j)]
              *exp(-(tab[(idx)*M+idy-j+y_small]-tab[idx*M+idy])
              *(tab[(idx)*M+idy-j+y_small]-tab[idx*M+idy])/var)
            );
            float r = (filter[(s)*(2*s+1)+(s-j)]
              *exp(-(tab[(idx)*M+idy+j+y_big]-tab[idx*M+idy])
              *(tab[(idx)*M+idy+j+y_big]-tab[idx*M+idy])/var)
            );

            output[idx*M+idy] += (l*tab[(idx)*M+idy-j+y_small]);
            output[idx*M+idy] += (r*tab[(idx)*M+idy+j+y_big]);

            sum += l+r;
          }

          sum += tl+tr+bl+br;
        }
        sum += t+b;
      }
      output[idx*M+idy] /= sum;
    }
  }
}

void box_filter(float *filter, int size) {
  for (int i=0; i<2*size+1; i++){
    for (int j=0; j<2*size+1; j++){
      filter[i*(2*size+1)+j] = 1;
    }
  }
}

void gaussian_filter(float *filter, int size) {
  float s = (float) (2*size+1);

  for (int i=0; i<2*size+1; i++){
    for (int j=0; j<2*size+1; j++){
      filter[i*(2*size+1)+j] =  exp(-((i-size)*(i-size)+(j-size)*(j-size))/ (2*s*s));
    }
  }
}

void conv_GPU(float *tab, int N, int M, float *filter,
  int s, float *tab_filtered, int N_threads) {
  clock_t start, finish;
  double duration;
  start = clock();

  float *tab_GPU;
  float *output_GPU;
  float *filter_GPU;
  // Allocate vector in device memory
  cudaMalloc(&tab_GPU, N * M * sizeof(float));
  cudaMalloc(&output_GPU, N * M * sizeof(float));
  cudaMalloc(&filter_GPU, (2*s+1) * (2*s+1) * sizeof(float));
  // Copy vectors from host memory to device memory
  cudaMemcpy(tab_GPU, tab, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_GPU, tab, N * M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_GPU, filter,  (2*s+1) * (2*s+1) *
    sizeof(float), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(N_threads, N_threads, 1);
  dim3 blocksPerGrid((int) ceil(N / (float)threadsPerBlock.x),
            (int) ceil(M / (float)threadsPerBlock.y), 1);

  conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, M, filter_GPU, s, output_GPU);

  cudaMemcpy(tab_filtered, output_GPU, N * M * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaFree(tab_GPU);
  cudaFree(filter_GPU);
  cudaFree(output_GPU);

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f",duration);
}
