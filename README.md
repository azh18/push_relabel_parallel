# push_relabel_mpi
Some parallel version of push label algorithm

## Openmpi

in mpi_impl floder

run:
`mpiexec -n <number of processes> ./mpi_push_relabel <input file>`

## Pthread

in pthread_impl floder

run:
`./pthread_push_relabel <input file> <number of threads>`


## Cuda

in cuda_impl floder

run:
`./cuda_push_relabel <input file> <num of blocks per grid> <number of thread per block>`
