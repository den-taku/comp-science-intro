#include <stdio.h>
#include <mpi.h>

#define N 20000
#define K 20000
#define P 1001001011

inline int f(int x, int y){
  return (x + y) % P;
}

inline int g(int i, int j){
  if (j == 0) return 1;
  return 0;
}

int C[K], tmp[K];

int main(int argc, char **argv){
  int my_rank, num_proc;
  int n, k;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();

  for (k = 0; k < K; k++){
    C[k] = g(0, k);
  }
  
  for (n = 1; n < N; n++){
    for (k = 0; k < K; k++){
      tmp[k] = C[k];
    }
    
    C[0] = g(n, 0);
    for(k = 1; k < K; k++){
      C[k] = f(tmp[k-1], tmp[k]);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();

  if (my_rank == 0){
    printf("%d %d %d\n", C[K/4], C[K/3], C[K/2]);
    printf("time %f\n", t2-t1);
  }
  
  return 0;
}