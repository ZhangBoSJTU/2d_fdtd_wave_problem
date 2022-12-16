#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"

// double 性能会差一些，float 的精度已经足够
// Check error codes for CUDA functions
#define CHECK(call)                                                            \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
    }                                                                          \
  }

using namespace std;

#define IT_NUM 100 // !=1
#define c 2.236068
#define PI 3.1415926535897932
#define Lx 4 // length of X
#define Ly 2

// Block dimensions 调整影响性能
#define BDIMX 32
#define BDIMY 32

double **calloc2D(int x, int y);
void free2D(double **u, int x, int y);

double precise_solution(double x, double y, double total_time) {
  double res = 0.0;
  double coeff = 1.0f;
  // shoule be double loop
  for (int m = 1; m < 100; m += 2) {
    for (int n = 1; n < 100; n += 2) {
      coeff = 1.0 / (m * m * m * n * n * n);
      res += coeff *
             cos(total_time * sqrt(5) * PI / 4 * sqrt(m * m + 4 * n * n)) *
             sin(m * PI * x / 4) * sin(n * PI * y / 2);
    }
  }
  return 0.426050 * res;
}

void print_height(double *h, int Nx, int Ny) {
  for (int j = 0; j < (Ny+1); j++) {
    for (int i = 0; i < (Nx+1); i++) {
      cout << setiosflags(ios::left) << setw(8) << setprecision(10)
           << h[i * (Ny+1) + j] << " ";
    }
    cout << endl;
  }
  cout << endl;
}

__global__ void kernel_2dfd(double *h0, double *h1, double *h2, double dt,
                            double h, int Nx, int Ny) {

  unsigned int it = blockIdx.x * blockDim.x + threadIdx.x;

  if (it >= 1 && it <= IT_NUM - 1) {
    for (int i = 1; i <= Nx - 1; i++) {
      for (int j = 1; j <= Ny - 1; j++) {
        h2[i * Ny + j] =
            2 * h1[i * Ny + j] - h0[i * Ny + j] +
            (c * c * dt * dt / (h * h)) *
                (h1[(i + 1) * Ny + j] + h1[(i - 1) * Ny + j] +
                 h1[i * Ny + j + 1] + h1[i * Ny + j - 1] - 4 * h1[i * Ny + j]);
      }
    }
    // renew h1(t-1) with h1(t) and h1(t) with h1(t+1)
    for (int i = 1; i < Nx - 1; i++) {
      for (int j = 1; j <= Ny - 1; j++) {
        h0[i * Ny + j] = h1[i * Ny + j];
        h1[i * Ny + j] = h2[i * Ny + j];
      }
    }
  }
}

int main() {
  // Print out specs of the main GPU
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  printf("GPU0:\t%s\t%d.%d:\n", deviceProp.name, deviceProp.major,
         deviceProp.minor);
  printf("\t%lu GB:\t total Global memory (gmem)\n",
         deviceProp.totalGlobalMem / 1024 / 1024 / 1000);
  printf("\t%lu MB:\t total Constant memory (cmem)\n",
         deviceProp.totalConstMem / 1024);
  printf("\t%lu MB:\t total Shared memory per block (smem)\n",
         deviceProp.sharedMemPerBlock / 1024);
  printf("\t%d:\t total threads per block\n", deviceProp.maxThreadsPerBlock);
  printf("\t%d:\t total registers per block\n", deviceProp.regsPerBlock);
  printf("\t%d:\t warp size\n", deviceProp.warpSize);
  printf("\t%d x %d x %d:\t max dims of block\n", deviceProp.maxThreadsDim[0],
         deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("\t%d x %d x %d:\t max dims of grid\n", deviceProp.maxGridSize[0],
         deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  CHECK(cudaSetDevice(0));

  double dx = 0.1;
  double dy = dx;
  double h = dx;          // 离散在一个均匀的网格 dx = dy = h
  int Nx = ceil(Lx / dx); // Nx : 40
  int Ny = ceil(Ly / dy); // Ny : 20
  double *h0, *h1, *h2;   // wave hight 波高
  h0 = (double *)calloc((Nx + 1) * (Ny + 1),
                        sizeof(double)); // t(n-1) 分配数据在栈空间
  h1 = (double *)calloc((Nx + 1) * (Ny + 1), sizeof(double)); // t(n)
  h2 = (double *)calloc((Nx + 1) * (Ny + 1), sizeof(double)); // t(n+1)

  // init
  // set the edge lines of the array to nmber 0 and
  // fill others values into slots
  for (int i = 0; i <= Nx; i++) {
    h1[i * (Ny + 1)] = 0;
    h1[i * (Ny + 1) + Ny] = 0;
  }
  for (int j = 0; j <= Ny; j++) {
    h1[j] = 0;
    h1[Nx * (Ny + 1) + j] = 0;
  }

  for (int i = 1; i <= Nx; i++) {
    for (int j = 1; j <= Ny; j++) {
      double x = h * i;
      double y = h * j;
      h1[i * (Ny+1) + j] = 0.1 * (4 * x - x * x) * (2 * y - y * y);
    }
  }

  // time step size should satify: dt < h / c
  double dt = h / (2 * c);
  

  // init h0
  for (int i = 1; i <= Nx - 1; i++) {
    for (int j = 1; j <= Ny - 1; j++) {
      h0[i * (Ny + 1) + j]=
          (c * c * dt * dt / (2 * h * h)) *
              (h1[(i + 1) * (Ny + 1) + j] + h1[(i - 1) * (Ny + 1) + j] +
               h1[i * (Ny + 1) + j + 1] + h1[i * (Ny + 1) + j - 1] -
               4 * h1[i * (Ny + 1) + j]);
    }
  }

  print_height(h1,Nx,Ny);

  // interatioral time step
  size_t nbytes =
      (Nx + 1) * (Ny + 1) * sizeof(double); // bytes to store nx * ny
  double *d_h0, *d_h1, *d_h2;
  CHECK(cudaMalloc((void **)&d_h0, nbytes));
  CHECK(cudaMalloc((void **)&d_h1, nbytes));
  CHECK(cudaMalloc((void **)&d_h2, nbytes));
  CHECK(cudaMemset(d_h2, 0, nbytes));
  CHECK(cudaMemcpy(d_h0, h0, nbytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_h1, h1, nbytes, cudaMemcpyHostToDevice));

  // Setup CUDA run
  dim3 block(128);
  dim3 grid((IT_NUM + block.x - 1)  / block.x);
  
  kernel_2dfd<<<grid, block>>>(d_h0, d_h1, d_h2, dt, h, Nx, Ny);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d_h1, d_h1, nbytes, cudaMemcpyDeviceToHost));
  

  // print_height(h1, Nx, Ny);

  // get the highest point
  double hmax = 0;
  int imax = 0, jmax = 0;
  for (int i = 1; i < Nx - 1; i++) {
    for (int j = 1; j <= Ny - 1; j++) {
      if (fabs(hmax) < fabs(h1[i * Nx + j])) {
        hmax = h1[i * Nx + j];
        imax = i;
        jmax = j;
      }
    }
  }
  cout << "hmax is " << hmax << " imax is " << imax << " jmax is " << jmax
       << endl;

  // get lines values from numerical results
  int Lx1_idx = static_cast<int>(0.25 * Lx / h);
  int Lx2_idx = static_cast<int>(0.50 * Lx / h);
  int Lx3_idx = static_cast<int>(0.75 * Lx / h);

  double *h_Lx1_numaric = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx2_numaric = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx3_numaric = (double *)calloc(Ny + 1, sizeof(double));

  for (int i = 0; i <= Ny; i++) {
    h_Lx1_numaric[i] = h1[Lx1_idx * Nx + i];
    h_Lx2_numaric[i] = h1[Lx2_idx * Nx + i];
    h_Lx3_numaric[i] = h1[Lx3_idx * Nx + i];
  }

  // for (size_t i = 0; i < Ny; i++) {
  //   cout << h_Lx1_numaric[i] << " " << h_Lx2_numaric[i] << " "
  //        << h_Lx3_numaric[i] << endl;
  // }

  double *h_Lx1_precise = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx2_precise = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx3_precise = (double *)calloc(Ny + 1, sizeof(double));

  double total_time = IT_NUM * dt;
  for (int i = 0; i <= Ny; i++) {
    h_Lx1_precise[i] = precise_solution((0.25 * Lx), (i * h), total_time);
    h_Lx2_precise[i] = precise_solution((0.50 * Lx), (i * h), total_time);
    h_Lx3_precise[i] = precise_solution((0.75 * Lx), (i * h), total_time);
  }
  // for (size_t i = 0; i < Ny; i++) {
  //   cout << h_Lx1_precise[i] << " " << h_Lx1_precise[i] << " "
  //        << h_Lx1_precise[i] << endl;
  // }

  CHECK(cudaFree(d_h0));
  CHECK(cudaFree(d_h1));
  CHECK(cudaFree(d_h2));
  CHECK(cudaDeviceReset());

  free(h0);
  free(h1);
  free(h2);
  free(h_Lx1_numaric);
  free(h_Lx2_numaric);
  free(h_Lx3_numaric);
  free(h_Lx1_precise);
  free(h_Lx2_precise);
  free(h_Lx3_precise);
  
  return 0;
  // * /
}

double **calloc2D(int x, int y) {

  double **u = (double **)calloc(x, sizeof(double));

  if (u == NULL) {
    printf(" memory cannot be allocated inside function memalloc2D");
    exit(-1);
  }

  for (int i = 0; i < x; ++i) {
    u[i] = (double *)calloc(y, sizeof(double));
    if (u[i] == NULL) {
      printf(" memory cannot be allocated");
      printf("i=%d\n", i);
      exit(-1);
    }
  }
  return u;
}

void free2D(double **u, int x, int y) {
  for (int i = 0; i < x; ++i) {
    free(u[i]);
  }
  free(u);
}
