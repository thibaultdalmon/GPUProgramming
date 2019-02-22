#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

__global__ void conv(float *tab, int N, int M, int ndim,
  float *filter, int s, float *output);
__global__ void conv_bil(float *tab, int N, int M, int ndim,
  float *filter, int s, float *output, float r);

void box_filter(float *filter, int size, int ndim);
void gaussian_filter(float *filter, int size, int ndim);

void conv_GPU(float *tab, int ndim, int *dim, float *filter,
  int s, float *tab_filtered, int N_threads);

typedef struct {
  float *img;
  int ndim;
  int *dim;
  int max_pix;
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
  int dim_dot = 1;
  float *filter;
  for (int d=0; d<image->ndim; d++){
    dim_dot*=image->dim[d];
  }
  if (image->ndim==2)
    filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
  if (image->ndim==3)
    filter = (float*) malloc((2*s+1)*(2*s+1)*3*sizeof(float));
  float * img_filtered = (float*) malloc(dim_dot*sizeof(float));
  box_filter(filter, s, image->ndim);
  conv_GPU(image->img, image->ndim, image->dim,
    filter, s, img_filtered, 16);
  printf("\n");
  image->img = img_filtered;
}

void gaussian_filtering(Image *image, int s){
  int dim_dot = 1;
  float *filter;
  for (int d=0; d<image->ndim; d++){
    dim_dot*=image->dim[d];
  }
  if (image->ndim==2)
    filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
  if (image->ndim==3)
    filter = (float*) malloc((2*s+1)*(2*s+1)*3*sizeof(float));
  float * img_filtered = (float*) malloc(dim_dot*sizeof(float));
  gaussian_filter(filter, s, image->ndim);
  conv_GPU(image->img, image->ndim, image->dim,
    filter, s, img_filtered, 16);
  printf("\n");
  image->img = img_filtered;
}

void bilateral_filtering(Image *image, int s, float r){
  clock_t start, finish;
  double duration;
  start = clock();

  float * filter;
  float * img_filtered;
  int N;
  int M;
  int N_threads;
  float size;

  if (image->ndim==2){
    filter = (float*) malloc((2*s+1)*(2*s+1)*sizeof(float));
    img_filtered = (float*) malloc(image->dim[0]*image->dim[1]
      *sizeof(float));
    N = image->dim[0];
    M = image->dim[1];
    N_threads = 16;
    size = (float) (2*s+1);

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

    conv_bil<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, M, image->ndim, filter_GPU, s, output_GPU, r);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Error!\n" );

    cudaMemcpy(img_filtered, output_GPU, N * M * sizeof(float),
      cudaMemcpyDeviceToHost);
    cudaFree(tab_GPU);
    cudaFree(filter_GPU);
    cudaFree(output_GPU);
  }

  if (image->ndim==3){
    filter = (float*) malloc((2*s+1)*(2*s+1)*3*sizeof(float));
    img_filtered = (float*) malloc(image->dim[0]*image->dim[1]
      *3*sizeof(float));
    N = image->dim[0];
    M = image->dim[1];
    N_threads = 16;
    size = (float) (2*s+1);

    for (int k=0; k<3; k++){
      for (int i=0; i<2*s+1; i++){
        for (int j=0; j<2*s+1; j++){
          filter[(i*(2*s+1)+j)*3+k] =  exp(-((i-s)*(i-s)+
          (j-s)*(j-s)+(k-1)*(k-1))/ (2*size*size*3));
        }
      }
    }

    float *tab_GPU;
    float *output_GPU;
    float *filter_GPU;
    // Allocate vector in device memory
    cudaMalloc(&tab_GPU, N * M * 3 *sizeof(float));
    cudaMalloc(&output_GPU, N * M * 3 *sizeof(float));
    cudaMalloc(&filter_GPU, (2*s+1) * (2*s+1) * 3 *sizeof(float));
    // Copy vectors from host memory to device memory
    cudaMemcpy(tab_GPU, image->img, N * M * 3 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_GPU, img_filtered, N * M * 3 *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_GPU, filter,  (2*s+1) * (2*s+1) * 3 *
      sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N_threads, N_threads, 3);
    dim3 blocksPerGrid((int) ceil(N / (float)threadsPerBlock.x),
              (int) ceil(M / (float)threadsPerBlock.y), 1);

    conv_bil<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, N, M, image->ndim, filter_GPU, s, output_GPU, r);
    if ( cudaSuccess != cudaGetLastError() )
      printf( "Error!\n" );

    cudaMemcpy(img_filtered, output_GPU, N * M * 3 * sizeof(float),
      cudaMemcpyDeviceToHost);
    cudaFree(tab_GPU);
    cudaFree(filter_GPU);
    cudaFree(output_GPU);
  }

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f\n",duration);
  image->img = img_filtered;
}

// TODO: case "P6"
void open_img(Image *image){
  FILE *fp;
  int height, width, max_pix;
  int c;

  fp = fopen("image_256x256.pgm", "r");
  fscanf(fp, "P%d\n", &c);
  switch (c) {
    case 2:
      fscanf(fp, "%d %d\n", &height, &width);
      fscanf(fp, "%d\n", &max_pix);

      image->ndim = 2;
      image->img = (float *) malloc(height*width*sizeof(float));
      image->dim = (int *) malloc(image->ndim*sizeof(int));
      image->dim[0] = height;
      image->dim[1] = width;
      image->max_pix = max_pix;

      for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
          fscanf(fp, "%d\n", &c);
          *(image->img+i*width+j) = (float) c;
        }
      }
      fclose(fp);
      break;
    case 6:
      fscanf(fp, "%d %d\n", &height, &width);
      fscanf(fp, "%d\n", &max_pix);

      image->ndim = 3;
      image->img = (float *) malloc(height*width*3*sizeof(float));
      image->dim = (int *) malloc(image->ndim*sizeof(int));
      image->dim[0] = height;
      image->dim[1] = width;
      image->dim[2] = 3;
      image->max_pix = max_pix;

      for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
          fscanf(fp, "%d\n", &c);
          *(image->img+i*width+j) = (float) c;
        }
      }
      fclose(fp);
  }
}

// TODO: case "P6"
void write_img(Image *image, int file_idx){
  FILE *fp;
  int height = image->dim[0];
  int width = image->dim[1];
  int max_pix = image->max_pix;

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

  fprintf(fp, "P2\n%d %d\n%d\n", height, width, max_pix);

  for (int i=0; i<height; i++){
    for (int j=0; j<width; j++)
      fprintf(fp, "%d\n", (int) *(image->img+i*width+j));
  }
  fclose(fp);
}

__global__ void conv(float *tab, int N, int M, int ndim,
  float *filter, int s, float *output){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ndim==2){
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
  if (ndim==3){
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx<N){
      if (idy<M){
        if (idz<3){
          float sum = 0;
          output[(idx*M+idy)*3+idz] = 0;
          for (int k=0; k<3; k++){
            output[(idx*M+idy)*3+idz] += filter[s*3+k+idz+1]*tab[(idx*M+idy)*3+(k+idz)%3];
            sum += filter[s*3+k+idz+1];
            for (int i=1; i<s+1; i++){
              int x_big = -N*(idx+i>=N);
              int x_small = N*(idx-i<0);
              float t = (filter[((s+i)*(2*s+1)+s)*3+(k+idz+1)%3]);
              float b = (filter[((s-i)*(2*s+1)+s)*3+(k+idz+1)%3]);

              output[(idx*M+idy)*3+idz] += (t*tab[((idx+i+x_big)*M)*3+(k+idz)%3]);
              output[(idx*M+idy)*3+idz] += (b*tab[((idx-i+x_small)*M)*3+(k+idz)%3]);

              for (int j=1; j<s+1; j++){
                int y_big = -M*(idy+j>=M);
                int y_small = M*(idy-j<0);
                float tl = (filter[((s+i)*(2*s+1)+(s+j))*3+(k+idz+1)%3]);
                float tr = (filter[((s+i)*(2*s+1)+(s-j))*3+(k+idz+1)%3]);
                float bl = (filter[((s-i)*(2*s+1)+(s+j))*3+(k+idz+1)%3]);
                float br = (filter[((s-i)*(2*s+1)+(s-j))*3+(k+idz+1)%3]);

                output[(idx*M+idy)*3+idz] += (tl*tab[((idx-i+x_small)*M+idy-j+y_small)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (tr*tab[((idx-i+x_small)*M+idy+j+y_big)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (bl*tab[((idx+i+x_big)*M+idy-j+y_small)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (br*tab[((idx+i+x_big)*M+idy+j+y_big)*3+(k+idz)%3]);

                if (i==1) {
                  float l = (filter[((s)*(2*s+1)+(s+j))*3+(k+idz+1)%3]);
                  float r = (filter[((s)*(2*s+1)+(s-j))*3+(k+idz+1)%3]);

                  output[(idx*M+idy)*3+idz] += (l*tab[((idx)*M+idy-j+y_small)*3+(k+idz)%3]);
                  output[(idx*M+idy)*3+idz] += (r*tab[((idx)*M+idy+j+y_big)*3+(k+idz)%3]);

                  sum += l+r;
                }

                sum += tl+tr+bl+br;
              }
              sum += t+b;
            }
          }
          output[(idx*M+idy)*3+idz] /= sum;
        }
      }
    }
  }
}

__global__ void conv_bil(float *tab, int N, int M, int ndim,
  float *filter, int s, float *output, float r){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ndim==2){
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

  if (ndim==3){
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    float var = 2*r*r*3;

    if (idx<=N){
      if (idy<=M){
        if (idz<3){
          output[(idx*M+idy)*3+idz] = 0;
          float sum = 0;
          for (int k=0; k<3; k++){
            output[(idx*M+idy)*3+idz] += filter[s*3+k+idz+1]*tab[(idx*M+idy)*3+(k+idz)%3];
            sum += filter[s];
            for (int i=1; i<s+1; i++){
              int x_big = -N*(idx+i>=N);
              int x_small = N*(idx-i<0);
              float t = (filter[((s+i)*(2*s+1)+s)*3+k+idz+1]
                *exp(-(tab[((idx+i+x_big)*M)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                *(tab[((idx+i+x_big)*M)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
              );
              float b = (filter[((s-i)*(2*s+1)+s)*3+k+idz+1]
                *exp(-(tab[((idx-i+x_small)*M)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                *(tab[((idx-i+x_small)*M)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
              );

              output[(idx*M+idy)*3+idz] += (t*tab[((idx+i+x_big)*M)*3+(k+idz)%3]);
              output[(idx*M+idy)*3+idz] += (b*tab[((idx-i+x_small)*M)*3+(k+idz)%3]);

              for (int j=1; j<s+1; j++){
                int y_big = -M*(idy+j>=M);
                int y_small = M*(idy-j<0);
                float tl = (filter[((s+i)*(2*s+1)+(s+j))*3+k+idz+1]
                  *exp(-(tab[((idx-i+x_small)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                  *(tab[((idx-i+x_small)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
                );
                float tr = (filter[((s+i)*(2*s+1)+(s-j))*3+k+idz+1]
                  *exp(-(tab[((idx-i+x_small)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                  *(tab[((idx-i+x_small)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
                );
                float bl = (filter[((s-i)*(2*s+1)+(s+j))*3+k+idz+1]
                  *exp(-(tab[((idx+i+x_big)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                  *(tab[((idx+i+x_big)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
                );
                float br = (filter[((s-i)*(2*s+1)+(s-j))*3+k+idz+1]
                  *exp(-(tab[((idx+i+x_big)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                  *(tab[((idx+i+x_big)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)]*3+(k+idz)%3)/var)
                );

                output[(idx*M+idy)*3+idz] += (tl*tab[((idx-i+x_small)*M+idy-j+y_small)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (tr*tab[((idx-i+x_small)*M+idy+j+y_big)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (bl*tab[((idx+i+x_big)*M+idy-j+y_small)*3+(k+idz)%3]);
                output[(idx*M+idy)*3+idz] += (br*tab[((idx+i+x_big)*M+idy+j+y_big)*3+(k+idz)%3]);

                if (i==1) {
                  float l = (filter[((s)*(2*s+1)+(s+j))*3+k+idz+1]
                    *exp(-(tab[((idx)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                    *(tab[((idx)*M+idy-j+y_small)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
                  );
                  float r = (filter[((s)*(2*s+1)+(s-j))*3+k+idz+1]
                    *exp(-(tab[((idx)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])
                    *(tab[((idx)*M+idy+j+y_big)*3+(k+idz)%3]-tab[(idx*M+idy)*3+(k+idz)%3])/var)
                  );

                  output[(idx*M+idy)*3+idz] += (l*tab[((idx)*M+idy-j+y_small)*3+(k+idz)%3]);
                  output[(idx*M+idy)*3+idz] += (r*tab[((idx)*M+idy+j+y_big)*3+(k+idz)%3]);

                  sum += l+r;
                }
                sum += tl+tr+bl+br;
              }
              sum += t+b;
            }
          }
          output[(idx*M+idy)*3+idz] /= sum;
        }
      }
    }
  }
}

void box_filter(float *filter, int size, int ndim) {
  if (ndim==2){
    for (int i=0; i<2*size+1; i++){
      for (int j=0; j<2*size+1; j++){
        filter[i*(2*size+1)+j] = 1;
      }
    }
  }
  if (ndim==3){
    for (int i=0; i<2*size+1; i++){
      for (int j=0; j<2*size+1; j++){
        for (int k=0; k<3; k++){
          filter[(i*(2*size+1)+j)*3+k] = 1;
        }
      }
    }
  }
}

void gaussian_filter(float *filter, int size, int ndim) {
  float s = (float) (2*size+1);
  if (ndim==2){
    for (int i=0; i<2*size+1; i++){
      for (int j=0; j<2*size+1; j++){
        filter[i*(2*size+1)+j] =  exp(-((i-size)*(i-size)+(j-size)*(j-size))/ (2*s*s));
      }
    }
  }
  if (ndim==3){
    for (int i=0; i<2*size+1; i++){
      for (int j=0; j<2*size+1; j++){
        for (int k=0; k<3; k++){
          float norm = (i-size)*(i-size)+(j-size)*(j-size)+(k-1)*(k-1);
          filter[(i*(2*size+1)+j)*3+k] =  exp(-norm/(2*s*s*3));
        }
      }
    }
  }
}

void conv_GPU(float *tab, int ndim, int *dim, float *filter,
  int s, float *tab_filtered, int N_threads) {
  clock_t start, finish;
  double duration;
  start = clock();

  int dim_dot = 1;
  for (int i=0; i<ndim; i++){
    dim_dot*=dim[i];
  }

  if (ndim==2){
    float *tab_GPU;
    float *output_GPU;
    float *filter_GPU;
    // Allocate vector in device memory
    cudaMalloc(&tab_GPU, dim_dot*sizeof(float));
    cudaMalloc(&output_GPU, dim_dot*sizeof(float));
    cudaMalloc(&filter_GPU, (2*s+1) * (2*s+1) * sizeof(float));
    // Copy vectors from host memory to device memory
    cudaMemcpy(tab_GPU, tab, dim_dot*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_GPU, tab, dim_dot*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_GPU, filter,  (2*s+1) * (2*s+1) *
      sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N_threads, N_threads, 1);
    dim3 blocksPerGrid((int) ceil(dim[0] / (float)threadsPerBlock.x),
              (int) ceil(dim[1] / (float)threadsPerBlock.y), 1);

    conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, dim[0], dim[1], ndim, filter_GPU, s, output_GPU);

    cudaMemcpy(tab_filtered, output_GPU, dim_dot*sizeof(float),
      cudaMemcpyDeviceToHost);
    cudaFree(tab_GPU);
    cudaFree(filter_GPU);
    cudaFree(output_GPU);
  }
  if (ndim==3){
    float *tab_GPU;
    float *output_GPU;
    float *filter_GPU;
    // Allocate vector in device memory
    cudaMalloc(&tab_GPU, dim_dot*sizeof(float));
    cudaMalloc(&output_GPU, dim_dot*sizeof(float));
    cudaMalloc(&filter_GPU, (2*s+1) * (2*s+1) * 3 * sizeof(float));
    // Copy vectors from host memory to device memory
    cudaMemcpy(tab_GPU, tab, dim_dot*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_GPU, tab, dim_dot*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_GPU, filter,  (2*s+1) * (2*s+1) * 3 *
      sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N_threads, N_threads, 3);
    dim3 blocksPerGrid((int) ceil(dim[0] / (float)threadsPerBlock.x),
              (int) ceil(dim[1] / (float)threadsPerBlock.y), 1);

    conv<<<blocksPerGrid,threadsPerBlock>>>(tab_GPU, dim[0], dim[1], ndim, filter_GPU, s, output_GPU);

    cudaMemcpy(tab_filtered, output_GPU, dim_dot*sizeof(float),
      cudaMemcpyDeviceToHost);
    cudaFree(tab_GPU);
    cudaFree(filter_GPU);
    cudaFree(output_GPU);
  }

  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  printf("%f",duration);
}
