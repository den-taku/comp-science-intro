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
  int n, k, p;
  double t1, t2;
  long long myL, myR;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  MPI_Barrier(MPI_COMM_WORLD);
  t1 = MPI_Wtime();


  myL = K / num_proc * my_rank;
  myR = myL + K / num_proc;
  myR = myR > K ? K : myR;

  for (k = myL; k < myR; k++){
    C[k] = g(0, k);
  }

  if (my_rank != num_proc - 1) {
    MPI_Send(&tmp[myR - 1], 1, MPI_INT, my_rank + 1, 0, MPI_COMM_WORLD);
  }
  
  for (n = 1; n < N; n++){
    for (k = myL; k < myR; k++){
      tmp[k] = C[k];
    }
    
    C[0] = g(n, 0);
    if (my_rank != 0) {
      MPI_Recv(&tmp[myL-1], 1, MPI_INT, my_rank - 1, n-1, MPI_COMM_WORLD, &status);
    }

    for(k = myL; k < myR; k++){
      C[k] = f(tmp[k-1], tmp[k]);
    }

    if (my_rank != num_proc - 1) {
      MPI_Send(&tmp[myR - 1], 1, MPI_INT, my_rank + 1, n, MPI_COMM_WORLD);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t2 = MPI_Wtime();

  if (my_rank == 0){
    for (p = 1; p < num_proc; ++p) {
      myL = K / num_proc * p;
      myR = myL + K / num_proc;
      myR = myR > K ? K : myR;
      MPI_Recv(&C[myL], myR - myL + 1, MPI_INT, p, N, MPI_COMM_WORLD, &status);
    }
    printf("%d %d %d\n", C[K/4], C[K/3], C[K/2]);
    printf("time %f\n", t2-t1);
  } else {
    MPI_Send(&C[myL], myR - myL + 1, MPI_INT, 0, N, MPI_COMM_WORLD);
  }
  
  return 0;
}