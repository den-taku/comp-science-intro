#include<stdio.h>
#include<omp.h>

#define N 10000000000LL

int main(){

  long long k;
  double sum = 0.0;
  double t1, t2;

  t1 = omp_get_wtime();

  for(k = N-1; k > 0; k -= 2){
    sum -= 1.0 / (2*k+1);
    sum += 1.0 / (2*k-1);
  }

  sum *= 4.0;

  t2 = omp_get_wtime();

  printf("%.12f\n", sum);
  printf("time: %f [s]\n", t2-t1);
  return 0;
}