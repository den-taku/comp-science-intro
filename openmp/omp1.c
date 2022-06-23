#include<stdio.h>
#include<omp.h>

#define N 20000
#define K 20000
#define P 1001001011

int C[N][K];

int main(){
  int n, k;
  double t1, t2;

  t1 = omp_get_wtime();
  
  for (n = 0; n < N; n++){
    C[n][0] = 1;
  }

  for (k = 1; k < K; k++){
    C[0][k] = 0;
  }

  for (n = 1; n < N; n++){
    for(k = 1; k < K; k++){
      C[n][k] = (C[n-1][k-1] + (long long) C[n-1][k]) % P;
    }
  }

  t2 = omp_get_wtime();

  printf("%d %d %d\n", C[N-1][K/4], C[N-1][K/3], C[N-1][K/2]);
  printf("time %f\n", t2-t1);
  
  return 0;
}