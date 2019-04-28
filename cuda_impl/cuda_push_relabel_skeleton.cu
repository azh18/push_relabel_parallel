/**
 * Name: ZHANG Bowen
 * Student id: 20552982
 * ITSC email: bzhangba@connect.ust.hk
 */

#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <chrono>

#include <vector>
#include <iostream>

#include "cuda_push_relabel.h"

using namespace std::chrono;

// #define DEBUG

#ifdef DEBUG
#define print_array(x, l) cout << ""; \
    cout << #x << ":{"; \
    for(int i=0;i<l;i++){ \
        cout << x[i] << ", "; \
    } \
    cout << "}" << endl;
#define pause sleep(1);
#else
#define print_array(x, l) 1;
#define pause sleep(0);
#endif

#define MAX_N 520 // alter this if you use larger N!!!

using namespace std;

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

/*
 *  You can add helper functions and variables as you wish.
*/

__constant__ int active_nodes_gpu[MAX_N];

__device__ inline int min_dev(int64_t a64, int b32){
    int64_t b64 = (int64_t)b32;
    int64_t result = a64<b64?a64:b64;
    return (int)result;
}

__device__ inline void atomicAdd(int64_t *addr, int64_t val){
    unsigned long long int assumed;
    unsigned long long int old = (unsigned long long int)(*addr);
    do{
        assumed = old;
        old = atomicCAS((unsigned long long int*)(addr), assumed, ((unsigned long long int)(*addr) + (unsigned long long int)val));
    } while(assumed != old);
}

/*
input: cap, flow, dist, excess, active_nodes

output: updated excess vector, stash_excess vector, updated flow matrix
*/
// v1: combine two loops in stage 1 together (in v2 we can use only one block for sub excess, try?)
__global__ void stage_1_kernel_v1(int *cap, int *flow, int *dist, int64_t *excess, int64_t *stash_excess, int n_active_nodes, int blocks_per_grid, int threads_per_block, int N){
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int uidx_start = bid;
    int uidx_cnt = n_active_nodes / blocks_per_grid;
    if(bid < n_active_nodes % blocks_per_grid){
        uidx_cnt += 1;
    }

    int vidx_start = tid;
    int vidx_cnt = N / threads_per_block;
    if(tid < N % threads_per_block){
        vidx_cnt += 1;
    }

    // load dist, own excess in local memory
    __shared__ int64_t own_excess;
    __shared__ int local_dist[MAX_N];
    __shared__ int stash_send[MAX_N];
    __shared__ int local_residual_cap[MAX_N];

    for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
        local_dist[v] = dist[v];
        stash_send[v] = 0;
    }

    for(int uidx = uidx_start, ucnt = 0; ucnt < uidx_cnt; ucnt++, uidx += blocks_per_grid){
        int u = active_nodes_gpu[uidx];
        // the following is to do with this u:

        for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
            local_residual_cap[v] = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
        }
        __syncthreads();

        // use single thread to sub excess, avoid smaller than 0
        if(tid == 0){
            own_excess = excess[u];
            for(int v = 0; v < N; v++){
                int residual_cap = local_residual_cap[v];
                if (residual_cap > 0 && local_dist[u] > local_dist[v] && own_excess > 0){
                    stash_send[v] = min_dev(own_excess, residual_cap);
                    own_excess -= stash_send[v];
                }
            }
        }
        __syncthreads();

        // use all threads to update flow
        for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
            if (stash_send[v] > 0){
                int this_stash_send = stash_send[v];
                atomicAdd(&flow[utils::dev_idx(u, v, N)], this_stash_send);
                atomicSub(&flow[utils::dev_idx(v, u, N)], this_stash_send);
                atomicAdd(&stash_excess[v], (int64_t)this_stash_send);
                stash_send[v] = 0;
            }
        }

        // write back data in local mem to global mem
        __syncthreads();
        excess[u] = own_excess;
    }
}

/*
process excess and stash_excess on GPU to avoid data transfer
*/
__global__ void update_excess(int64_t *excess, int64_t *stash_excess, int blocks_per_grid, int threads_per_block, int N){
    int tid = threads_per_block * blockIdx.x + threadIdx.x;
    int n_total_threads = threads_per_block * blocks_per_grid;
	int vidx_start = tid;
	int vidx_cnt = N / n_total_threads;
	if (tid < N % n_total_threads) {
		vidx_cnt += 1;
    }
    for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += n_total_threads){
        excess[v] += stash_excess[v];
        stash_excess[v] = 0;
    }
}

/*
input: flow matrix, stash_dist, dist, active_nodes, cap, flow

output: updated flow matrix, updated stash_dist vector
*/
__global__ void stage_2_kernel(int *cap, int *flow, int64_t* excess, int *dist, int* stash_dist, int n_active_nodes, int blocks_per_grid, int threads_per_block, int N){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uidx_start = bid;
    int uidx_cnt = n_active_nodes / blocks_per_grid;
    if(bid < n_active_nodes % blocks_per_grid){
        uidx_cnt += 1;
    }

    int vidx_start = tid;
    int vidx_cnt = N / threads_per_block;
    if(tid < N % threads_per_block){
        vidx_cnt += 1;
    }
    // get min_dist for each u
    for(int uidx = uidx_start, ucnt = 0; ucnt < uidx_cnt; ucnt++, uidx += blocks_per_grid){
        int u = active_nodes_gpu[uidx];
        __shared__ int min_dist;
        if(tid == 0){
            min_dist = INT32_MAX;
        }
		__syncthreads();
        if(excess[u] > 0){
            for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
                int residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0) {
                    atomicMin(&min_dist, dist[v]);
					stash_dist[u] = min_dist + 1;
                }
            }
        }
        __syncthreads();
        if(tid == 0){
            dist[u] = stash_dist[u];
        }
        __syncthreads();
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
/*
    *  Please fill in your codes here.
    */
    if(N > MAX_N){
        printf("Please set macro MAX_N(default 520) in \"cuda_push_relabel_skeleton.cu\" larger than N(now %d)!!!\n", N);
        throw("");
    }

    // do pre-flow on CPU
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    vector<int> active_nodes;
    int *stash_send = (int *) calloc(N * N, sizeof(int));
    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    // alloc mem on GPU and transfer data (how about zero-copy?)
    int *cap_gpu, *flow_gpu, *dist_gpu, *stash_dist_gpu;
    int64_t *excess_gpu, *stash_excess_gpu;
    GPUErrChk(cudaMalloc(&cap_gpu, N*N*sizeof(int)));
    GPUErrChk(cudaMalloc(&flow_gpu, N*N*sizeof(int)));
    GPUErrChk(cudaMalloc(&dist_gpu, N*sizeof(int)));
    GPUErrChk(cudaMalloc(&stash_dist_gpu, N*sizeof(int)));
    GPUErrChk(cudaMalloc(&excess_gpu, N*sizeof(int64_t)));
    GPUErrChk(cudaMalloc(&stash_excess_gpu, N*sizeof(int64_t)));
    // transfer before main loop: cap, flow, dist, excess
    GPUErrChk(cudaMemcpy(cap_gpu, cap, N*N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(flow_gpu, flow, N*N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(dist_gpu, dist, N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(excess_gpu, excess, N*sizeof(int64_t), cudaMemcpyHostToDevice));

    while(!active_nodes.empty()){
        int n_active_nodes = active_nodes.size();
        GPUErrChk(cudaMemcpyToSymbol(active_nodes_gpu, active_nodes.data(), n_active_nodes*sizeof(int)));

        // stage 1 kernel, output: updated excess vector, stash_excess vector, updated flow matrix
        int n_block_use = n_active_nodes;
        if(blocks_per_grid < n_block_use){
            n_block_use = blocks_per_grid;
        }
        stage_1_kernel_v1<<<n_block_use, threads_per_block>>>(cap_gpu, flow_gpu, dist_gpu, excess_gpu, stash_excess_gpu, n_active_nodes, n_block_use, threads_per_block, N);

        // Stage 2: relabel (update dist to stash_dist and finally to dist).
        stage_2_kernel<<<n_block_use, threads_per_block>>>(cap_gpu, flow_gpu, excess_gpu, dist_gpu, stash_dist_gpu, n_active_nodes, n_block_use, threads_per_block, N);

        // Stage 3: apply excess-flow changes for destination vertices.
        n_block_use = N / threads_per_block + 1;
        update_excess<<<n_block_use, threads_per_block>>>(excess_gpu, stash_excess_gpu, n_block_use, threads_per_block, N);
        GPUErrChk(cudaMemcpy(excess, excess_gpu, N*sizeof(int64_t), cudaMemcpyDeviceToHost));

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }
    GPUErrChk(cudaMemcpy(flow, flow_gpu, N*N*sizeof(int), cudaMemcpyDeviceToHost));
    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    GPUErrChk(cudaFree(cap_gpu));
    GPUErrChk(cudaFree(flow_gpu));
    GPUErrChk(cudaFree(dist_gpu));
    GPUErrChk(cudaFree(stash_dist_gpu));
    GPUErrChk(cudaFree(excess_gpu));
    GPUErrChk(cudaFree(stash_excess_gpu));

    return 0;
}
