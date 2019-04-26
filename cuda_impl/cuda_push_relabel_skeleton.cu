/**
 * Name:
 * Student id:
 * ITSC email:
 */

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>

#include "cuda_push_relabel.h"

#define MAX_N 520

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

__device__ inline int min_dev(int64_t a64, int b32){
    int a32 = (int)a64;
    return a32<b32? a32 : b32 ;
}

__device__ inline atomicAdd(int64_t *addr, int64_t val){
    unsigned long long int assumed;
    unsigned long long int old = (unsigned long long int)(*addr);
    do{
        assumed = old;
        old = atomicCAS(addr, assumed, (int64_t)((unsigned long long int)(*addr) + (unsigned long long int)val));
    } while(assumed != old);
}

/*
input: 
*/

/*
input: cap, flow, dist, excess, active_nodes

each block handle a U, threads in a block compute stash_send in parallel, and record excess u in local memory (atomic add)

write flow and inverseFlow matrix seperately, finally add them up as the new flow matrix
stash_excess is added inside each block in parallel in local mem

sum up stash_excess among blocks finally using atomic add to global mem

output: updated excess vector, stash_excess vector, updated flow/inverseFlow matrix
*/
// v1: combine two loops in stage 1 together (in v2 we can use only one block for sub excess, try?)
__global__ void stage_1_kernel_v1(int *cap, int *flow, int *inverseFlow, int *dist, int64_t *excess, int64_t *stash_excess, int* active_nodes, int n_active_nodes, int blocks_per_grid, int threads_per_block, int N){
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
    __shared__ bool open_v[MAX_N];
    __shared__ int stash_send[MAX_N];

    for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
        local_dist[v] = dist[v];
    }

    for(int uidx = uidx_start, int ucnt = 0; ucnt < uidx_cnt; ucnt++, uidx += blocks_per_grid){
        int u = active_nodes[uidx];
        // the following is to do with this u:

        // flush open_v
        for(int v = vidx_start, vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
            open_v[v] = false;
        }

        // use single thread to sub excess, avoid smaller than 0
        if(tid == 0){
            own_excess = excess[u];
            for(int v = 0; v < N; v++){
                int residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0 && local_dist[u] > local_dist[v] && own_excess > 0){
                    stash_send[v] = min_dev(own_excess, residual_cap);
                    // excess[u] -= stash_send[utils::dev_idx(u, v, N)];
                    own_excess -= stash_send[v];
                    open_v[v] = true;
                }
            }
        }
        __synthreads();

        // use all threads to update flow/inverseFlow
        for(int v = vidx_start, int vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
            if (open_v[v]){
                int this_stash_send = stash_send[v];
                flow[utils::dev_idx(u, v, N)] += this_stash_send;
                inverseFlow[utils::dev_idx(v, u, N)] = -this_stash_send;
                // stash_excess[v] += stash_send;
                atomicAdd(&stash_excess[v], (int64_t)this_stash_send);
            }
        }

        // write back data in local mem to global mem
        __synthreads();
        excess[u] = own_excess;
    }
}

/*
input: flow/inverseFlow matrix, stash_dist, dist, active_nodes, cap, flow

firstly sum up flow and inverseFlow.

each block maintain own minimum stash_dist for each u, use atomic min

then copy local minimum stash_dist to cpu memory

output: updated flow matrix, updated stash_dist vector

*/
__global__ void stage_2_kernel(int *cap, int *flow, int* inverseFlow, int *dist, int* stash_dist, int* active_nodes, int n_active_nodes, int blocks_per_grid, int threads_per_block, int N){
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
    // update flow by inverseFlow
    for(int uidx = uidx_start, int ucnt = 0; ucnt < uidx_cnt; ucnt++, uidx += blocks_per_grid){
        int u = active_nodes[uidx];
        for(int v = vidx_start, int vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
            flow[utils::dev_idx(u, v, N)] += inverseFlow[utils::dev_idx(u, v, N)];
        }
    }
    __synthreads();
    // get min_dist for each u
    for(int uidx = uidx_start, int ucnt = 0; ucnt < uidx_cnt; ucnt++, uidx += blocks_per_grid){
        int u = active_nodes[uidx];
        if(excess[u] > 0){
            __shared__ int min_dist = INT32_MAX;
            for(int v = vidx_start, int vcnt = 0; vcnt < vidx_cnt; vcnt++, v += threads_per_block){
                int residual_cap = cap[utils::dev_idx(u, v, N)] - flow[utils::dev_idx(u, v, N)];
                if (residual_cap > 0) {
                    // min_dist = min(min_dist, dist[v]);
                    atomicMin(&min_dist, dist[v]);
                }
            }
            if(tid == 0){
                stash_dist[u] = min_dist + 1;
            }
        }
        __synthreads();
    }
}

int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow) {
/*
    *  Please fill in your codes here.
    */

    // do pre-flow on CPU
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));
    int* inverseFlow = (int*)calloc(N*N, sizeof(int));

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
    int *cap_gpu, *flow_gpu, *inverseFlow_gpu, *dist_gpu, *stash_dist_gpu, *active_nodes_gpu;
    int64_t *excess_gpu, *stash_excess_gpu;
    GPUErrChk(cudaMalloc(&cap_gpu, N*N*sizeof(int)));
    GPUErrChk(cudaMalloc(&flow_gpu, N*N*sizeof(int)));
    GPUErrChk(cudaMalloc(&inverseFlow_gpu, N*N*sizeof(int)));
    GPUErrChk(cudaMalloc(&dist_gpu, N*sizeof(int)));
    GPUErrChk(cudaMalloc(&stash_dist_gpu, N*sizeof(int)));
    GPUErrChk(cudaMalloc(&active_nodes_gpu, N*sizeof(int)));
    GPUErrChk(cudaMalloc(&excess_gpu, N*sizeof(int64_t)));
    GPUErrChk(cudaMalloc(&stash_excess_gpu, N*sizeof(int64_t)));
    // transfer before main loop: cap, flow, dist, excess
    // transfer in main loop(update): active_nodes, flow, inverseflow, dist, excess, stash_dist, stash_excess
    GPUErrChk(cudaMemcpy(cap_gpu, cap, N*N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(flow_gpu, flow, N*N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(dist_gpu, dist, N*sizeof(int), cudaMemcpyHostToDevice));
    GPUErrChk(cudaMemcpy(excess_gpu, excess, N*sizeof(int64_t), cudaMemcpyHostToDevice));


    while(!active_nodes.empty()){
        int n_active_nodes = active_nodes.size();
        GPUErrChk(cudaMemcpy(active_nodes_gpu, active_nodes.data(), n_active_nodes*sizeof(int), cudaMemcpyHostToDevice));
        GPUErrChk(cudaMemset(inverseFlow_gpu, 0, N*N*sizeof(int)));
        GPUErrChk(cudaMemset(stash_excess_gpu, 0, N*sizeof(int64_t)));

        // stage 1 kernel, output: updated excess vector, stash_excess vector, updated flow/inverseFlow matrix
        GPUErrChk(cudaMemcpy(excess_gpu, excess, N*sizeof(int64_t), cudaMemcpyHostToDevice));
        GPUErrChk(cudaMemcpy(dist_gpu, dist, N*sizeof(int), cudaMemcpyHostToDevice));
        stage_1_kernel_v1<<<blocks_per_grid, threads_per_block>>>(cap_gpu, flow_gpu, inverseFlow_gpu, dist_gpu, excess_gpu, stash_excess_gpu, active_nodes_gpu, n_active_nodes, blocks_per_grid, threads_per_block, N);
        cudaDeviceSynchronize();     
        // collect result if kernels do not share same division.
        // GPUErrChk(cudaMemcpy(excess, excess_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
        // GPUErrChk(cudaMemcpy(stash_excess, stash_excess_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
        // GPUErrChk(cudaMemcpy(flow, flow_gpu, N*N*sizeof(int), cudaMemcpyDeviceToHost));
        // GPUErrChk(cudaMemcpy(inverseFlow, inverseFlow_gpu, N*N*sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        // stage 2 kernel
        GPUErrChk(cudaMemcpy(dist_gpu, dist, N*sizeof(int64_t), cudaMemcpyHostToDevice));
        GPUErrChk(cudaMemcpy(stash_dist_gpu, stash_dist, N*sizeof(int64_t), cudaMemcpyHostToDevice));
        stage_2_kernel<<<blocks_per_grid, threads_per_block>>>(cap_gpu, flow_gpu, inverseFlow_gpu, dist_gpu, stash_dist_gpu, active_nodes_gpu, n_active_nodes, blocks_per_grid, threads_per_block, N);
        cudaDeviceSynchronize();
        // collect result.
        GPUErrChk(cudaMemcpy(excess, excess_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
        GPUErrChk(cudaMemcpy(stash_excess, stash_excess_gpu, N*sizeof(int), cudaMemcpyDeviceToHost));
        GPUErrChk(cudaMemcpy(flow, flow_gpu, N*N*sizeof(int), cudaMemcpyDeviceToHost));
        GPUErrChk(cudaMemcpy(inverseFlow, inverseFlow_gpu, N*N*sizeof(int), cudaMemcpyDeviceToHost));
        // Stage 3: update dist.
        swap(dist, stash_dist);

        // Stage 4: apply excess-flow changes for destination vertices.
        for (auto v = 0; v < N; v++) {
            if (stash_excess[v] != 0) {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
    }
    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);

    return 0;
}
