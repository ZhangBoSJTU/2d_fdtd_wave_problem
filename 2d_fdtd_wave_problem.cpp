#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define IT_NUM 100 // !=1
#define c 2.236068
#define PI 3.1415926535897932
#define Lx 4 // length of X
#define Ly 2

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

void print_height(double **h, int Nx, int Ny) {
  for (int j = 0; j <= Ny; j++) {
    for (int i = 0; i <= Nx; i++) {
      cout << setiosflags(ios::left) << setw(8) << setprecision(10) << h[i][j]
           << " ";
    }
    cout << endl;
  }
  cout << endl;
}

int main() {
  double dx = 0.1;
  double dy = dx;
  double h = dx;                 // 离散在一个均匀的网格 dx = dy = h
  int Nx = ceil(Lx / dx);        // Nx : 40
  int Ny = ceil(Ly / dy);        // Ny : 20
  double **h0, **h1, **h2;       // wave hight 波高
  h0 = calloc2D(Nx + 1, Ny + 1); // t(n-1) 分配数据在栈空间
  h1 = calloc2D(Nx + 1, Ny + 1); // t(n)
  h2 = calloc2D(Nx + 1, Ny + 1); // t(n+1)

  // init
  // set the edge lines of the array to nmber 0 and
  // fill others values into slots
  for (int i = 0; i <= Nx; i++) {
    h1[i][0] = 0;
    h1[i][Ny] = 0;
  }
  for (int j = 0; j <= Ny; j++) {
    h1[0][j] = 0;
    h1[Nx][j] = 0;
  }

  for (int i = 1; i <= Nx; i++) {
    for (int j = 1; j <= Ny; j++) {
      double x = h * i;
      double y = h * j;
      h1[i][j] = 0.1 * (4 * x - x * x) * (2 * y - y * y);
    }
  }

  // time step size should satify: dt < h / c
  double dt = h / (2 * c);

  

  // init h0
  for (int i = 1; i <= Nx - 1; i++) {
    for (int j = 1; j <= Ny - 1; j++) {
      h0[i][j] = h1[i][j] + (c * c * dt * dt / (2 * h * h)) *
                                (h1[i + 1][j] + h1[i - 1][j] + h1[i][j + 1] +
                                 h1[i][j - 1] - 4 * h1[i][j]);
    }
  }
  print_height(h1, Nx, Ny);

  // interatioral time step
  for (int it = 1; it <= IT_NUM - 1; it++) {
    for (int i = 1; i <= Nx - 1; i++) {
      for (int j = 1; j <= Ny - 1; j++) {
        h2[i][j] = 2 * h1[i][j] - h0[i][j] +
                   (c * c * dt * dt / (h * h)) *
                       (h1[i + 1][j] + h1[i - 1][j] + h1[i][j + 1] +
                        h1[i][j - 1] - 4 * h1[i][j]);
      }
    }
    // renew h1(t-1) with h1(t) and h1(t) with h1(t+1)
    for (int i = 1; i < Nx - 1; i++) {
      for (int j = 1; j <= Ny - 1; j++) {
        h0[i][j] = h1[i][j];
        h1[i][j] = h2[i][j];
      }
    }
  }

  print_height(h1, Nx, Ny);

  // get the highest point
  double hmax = 0;
  int imax = 0, jmax = 0;
  for (int i = 1; i < Nx - 1; i++) {
    for (int j = 1; j <= Ny - 1; j++) {
      if (fabs(hmax) < fabs(h1[i][j])) {
        hmax = h1[i][j];
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

  cout << Lx1_idx << " " << Lx2_idx << " " << Lx2_idx << endl;

  double *h_Lx1_numaric = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx2_numaric = (double *)calloc(Ny + 1, sizeof(double));
  double *h_Lx3_numaric = (double *)calloc(Ny + 1, sizeof(double));

  for (int i = 0; i <= Ny; i++) {
    h_Lx1_numaric[i] = h1[Lx1_idx][i];
    h_Lx2_numaric[i] = h1[Lx2_idx][i];
    h_Lx3_numaric[i] = h1[Lx3_idx][i];
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

  free2D(h0, Nx, Ny);
  free2D(h1, Nx, Ny);
  free2D(h2, Nx, Ny);
  free(h_Lx1_numaric);
  free(h_Lx2_numaric);
  free(h_Lx3_numaric);
  free(h_Lx1_precise);
  free(h_Lx2_precise);
  free(h_Lx3_precise);
  return 0;
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
