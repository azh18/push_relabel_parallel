/*
 * Do not change this file.
 * This is a CUDA version of the push-relabel algorithm
 * Compile: nvcc -std=c++11 -arch=compute_52 -code=sm_52 main.cu cuda_push_relabel_skeleton.cu -o cuda_push_relabel
 * Run: ./cuda_push_relabel <input file> <num of blocks per grid> <number of thread per block>
 */

#pragma once

/**
 * Push-relabel algorithm. Find the maximum-flow from vertex src to vertex sink.
 * @param blocks_per_grid number of blocks per grid
 * @param threads_per_block number of threads per block
 * @param N number of vertices
 * @param src src vertex of the maximum flow problem
 * @param sink sink vertex of the maximum flow problem
 * @param *cap capacity matrix (positive for each edge, zero for non-edge)
 * @param *flow the flow matrix
 * @attention we will use the flow matrix for the verification
*/
int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow);

#define GPUErrChk(ans) { utils::GPUAssert((ans), __FILE__, __LINE__); }

namespace utils {
    /*
     * translate 2-dimension coordinate to 1-dimension
     */
    int idx(int x, int y, int n);

    inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort)
                exit(code);
        }
    }

    inline __device__ int dev_idx(int x, int y, int n) {
        return x * n + y;
    }
}
