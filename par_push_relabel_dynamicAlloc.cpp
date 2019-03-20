/*
本实现中每次循环都按active_node重新平均划分任务，以实现负载均衡。
缺点是需要传递flow等信息，通讯开销较大。
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include <chrono>
#include "unistd.h"
#include "mpi_push_relabel.h"

// #define DEBUG
#ifdef DEBUG
#define print_array(x, l) cout << "[proc" << processRank << "]:"; \
    cout << #x << ":{"; \
    for(int i=0;i<l;i++){ \
        cout << x[i] << ", "; \
    } \
    cout << "}" << endl;
#else
#define print_array(x, l) 1;
#endif

#define pause sleep(0);

using namespace std;
using namespace std::chrono;

typedef struct InverseFlowPair{
    int src;
    int dst;
    int val;
}InverseFlowPair;

MPI_Datatype MpiInverseFlowPair;

void register_inverse_flow_pair_to_mpi(MPI_Datatype *MpiInverseFlowPair){
    InverseFlowPair templ;
    int blockLength[3] = {1,1,1};
    MPI_Aint disp[3] = {0, 0, 0};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint baseAddr, dstAddr, valAddr;
    MPI_Get_address(&templ.src, &baseAddr);
    MPI_Get_address(&templ.dst, &dstAddr);
    MPI_Get_address(&templ.val, &valAddr);
    disp[1] = dstAddr - baseAddr;
    disp[2] = valAddr - baseAddr;
    MPI_Type_create_struct(3, blockLength, disp, types, MpiInverseFlowPair);
    MPI_Type_commit(MpiInverseFlowPair);
}

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src)
{
    dist[src] = N;
    for (auto v = 0; v < N; v++)
    {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

inline int get_castv_cnt_disp_array(int nTotalWorks, int nWorkers, vector<int> &cnt, vector<int> &disp){
    cnt.clear(); disp.clear();
    int nBaseWorkload = nTotalWorks/nWorkers;
    int nHeavyProc = nTotalWorks%nWorkers;
    int offset = 0;
    for(int i=0;i<nWorkers;i++){
        disp.push_back(offset);
        if(i<nHeavyProc){
            offset += (nBaseWorkload+1);
            cnt.push_back(nBaseWorkload+1);
        } else {
            offset += nBaseWorkload;
            cnt.push_back(nBaseWorkload);
        }
    }
    return 0;
}

int sync_flow(int* flow, int N, int processRank, int nProcess, MPI_Comm comm, vector<InverseFlowPair> &outInverseFlows){
    // 发送outInverseFlow并更新各自flow
    static vector<int> nOutInverseFlows;
    nOutInverseFlows.resize(nProcess, 0);
    int localOutInverseFlows = outInverseFlows.size();
    // phase 1: 每个进程广播自己的修改个数
    MPI_Allgather(&localOutInverseFlows, 1, MPI_INT, nOutInverseFlows.data(), 1, MPI_INT, comm);

    // phase 2: 每个进程广播自己的修改内容
    vector<int> disp, cnt;
    disp.resize(nProcess);
    cnt.resize(nProcess);
    int offset = 0;
    for(int i=0; i<nProcess; i++){
        int n = nOutInverseFlows[i];
        cnt[i] = n;
        disp[i] = offset;
        offset += n;
    }

    static vector<InverseFlowPair> inInverseFlowBuffer;
    int needBufferSize = 0;
    for(int i=0;i<nProcess;i++){
        needBufferSize += nOutInverseFlows[i];
    }
    inInverseFlowBuffer.resize(needBufferSize);
    MPI_Allgatherv(outInverseFlows.data(), nOutInverseFlows[processRank], MpiInverseFlowPair, inInverseFlowBuffer.data(), cnt.data(), disp.data(), MpiInverseFlowPair, comm);

    // phase 3: 各自更新flow，保持一致
    for(auto ptr = inInverseFlowBuffer.begin(); ptr != inInverseFlowBuffer.end(); ptr++){
        // update inverse flow
        int src = ptr->src, dst = ptr->dst, val = ptr->val;
        flow[utils::idx(dst, src, N)] += val;
        flow[utils::idx(src, dst, N)] -= val;
    }
    return 0;
}

/*
sync dist array across processes
*/
int sync_dist(int *dist, const int *stash_dist, int N, int processRank, int nProcess, const vector<int> &activeNodes, const vector<int> &castvCnt, const vector<int> &castvDisp, MPI_Comm comm){
    static vector<int> inBuffer;
    inBuffer.resize(activeNodes.size());
    static vector<int> outBuffer;
    outBuffer.resize(castvCnt[processRank]);
    for(int i=0; i<castvCnt[processRank]; i++){
        int nodeIdx = castvDisp[processRank] + i;
        outBuffer[i] = stash_dist[activeNodes[nodeIdx]];
    }
    MPI_Allgatherv(outBuffer.data(), castvCnt[processRank], MPI_INT, inBuffer.data(), castvCnt.data(), castvDisp.data(), MPI_INT, comm);
    print_array(inBuffer, activeNodes.size());
    for(int i=0;i<activeNodes.size(); i++){
        dist[activeNodes[i]] = inBuffer[i];
    }
    return 0;
}

/*
sync excess array across processes
*/
int sync_excess(int64_t *excess, int N, int processRank, int nProcess, const vector<int> &activeNodes, const vector<int> &castvCnt, const vector<int> &castvDisp, MPI_Comm comm){
    static vector<int64_t> inBuffer;
    inBuffer.resize(activeNodes.size());
    static vector<int64_t> outBuffer;
    outBuffer.resize(castvCnt[processRank]);
    for(int i=0; i<castvCnt[processRank]; i++){
        int nodeIdx = castvDisp[processRank] + i;
        outBuffer[i] = excess[activeNodes[nodeIdx]];
    }
    MPI_Allgatherv(outBuffer.data(), castvCnt[processRank], MPI_INT64_T, inBuffer.data(), castvCnt.data(), castvDisp.data(), MPI_INT64_T, comm);
    for(int i=0;i<activeNodes.size(); i++){
        excess[activeNodes[i]] = inBuffer[i];
    }
    return 0;
}

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    register_inverse_flow_pair_to_mpi(&MpiInverseFlowPair);
    int nProcess=p, processRank=my_rank;
    // broadcast basic vars
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    MPI_Bcast(&src, 1, MPI_INT, 0, comm);
    MPI_Bcast(&sink, 1, MPI_INT, 0, comm);
    if (processRank != 0) {
        cap = (int *)calloc(N * N, sizeof(int));
        flow = (int *)calloc(N * N, sizeof(int));
    }
    MPI_Bcast(cap, N * N, MPI_INT, 0, comm);

    // pre-compute displacement&cnt of vertex for each process
    vector<int> CastvCnt, CastvDisp;

    // alloc mem for immediate vars
    int *dist = (int *)calloc(N, sizeof(int));
    int *stash_dist = (int *)calloc(N, sizeof(int));
    int64_t *excess = (int64_t *)calloc(N, sizeof(int64_t));
    int64_t *stash_excess = (int64_t *)calloc(N, sizeof(int64_t));
    int *stash_send = (int *)calloc(N * N, sizeof(int));

    // do first flow
    pre_flow(dist, excess, cap, flow, N, src);

    vector<int> globalActiveNodes;
    /*
    PStage 3:
    active nodes are managed by process 0
    */
    for (int u = 0; u < N; u++) {
        if (u != src && u != sink) {
            globalActiveNodes.emplace_back(u);
        }
    }
    int nGlobalActiveNodes = globalActiveNodes.size();
    print_array(cap, N*N);
    // make data ready use about 0.01s on dataset2, so it should not be the reason of slow
    // Four-Stage Pulses.
    while (nGlobalActiveNodes > 0) {
        // clear immediate vars
        memset(stash_excess, 0, sizeof(int64_t)*N);
        memset(stash_dist, 0, sizeof(int)*N);

        // divide tasks
        get_castv_cnt_disp_array(nGlobalActiveNodes, nProcess, CastvCnt, CastvDisp);
        int nWorkload = CastvCnt[processRank];
        int jobOffset = CastvDisp[processRank];
        print_array(CastvCnt, nProcess);
        print_array(CastvDisp, nProcess);
        print_array(globalActiveNodes, globalActiveNodes.size());

        /*
        Stage 1:

        Push, modify stash_send
        */

        for(int i=0;i<nWorkload;i++){
            int jobIdx = jobOffset + i;
            int u = globalActiveNodes[jobIdx];
            for (int v = 0; v < N; v++)
            {
                int residual_cap = cap[utils::idx(u, v, N)] -
                                   flow[utils::idx(u, v, N)];
                if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0)
                {
                    stash_send[utils::idx(u, v, N)] = std::min<int64_t>(excess[u], residual_cap);
                    excess[u] -= stash_send[utils::idx(u, v, N)];
                }
            }
        }

        // update modified excess
        sync_excess(excess, N, processRank, nProcess, globalActiveNodes, CastvCnt, CastvDisp, comm);
        print_array(stash_send, N*N);
        print_array(excess, N);


        /*
        update flow matrix and stash_excess according to stash_send
        */

        // update flow matrix on all processes
        vector<InverseFlowPair> outInverseFlows;
        for (int i=0;i<nWorkload;i++)
        {
            int jobIdx = jobOffset + i;
            int u = globalActiveNodes[jobIdx];
            for (int v = 0; v < N; v++)
            {
                if (stash_send[utils::idx(u, v, N)] > 0)
                {
                    // 这里如果按u划分，那么stash_send不必传送，如果按v划分，则stash_excess不必传送。考虑stash_send规模更大， 采用按u划分。
                    InverseFlowPair inverseFlow;
                    inverseFlow.dst = u;
                    inverseFlow.src = v;
                    inverseFlow.val = stash_send[utils::idx(u, v, N)];
                    outInverseFlows.emplace_back(inverseFlow);
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }

        // sync flow data across processes
        sync_flow(flow, N, processRank, nProcess, comm, outInverseFlows);
        // flow should be consistent after this

        print_array(flow, N*N);

        // sum up all stash_excess
        int64_t *new_stash_excess = (int64_t *)calloc(N, sizeof(int64_t));
        MPI_Allreduce(stash_excess, new_stash_excess, N, MPI_INT64_T, MPI_SUM, comm);
        memcpy(stash_excess, new_stash_excess, sizeof(int64_t)*N);
        free(new_stash_excess);
        // stash_excess should be consistent after this

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        for (int i=0;i<nWorkload;i++)
        {
            int jobIdx = jobOffset + i;
            int u = globalActiveNodes[jobIdx];
            if (excess[u] > 0)
            {
                int min_dist = INT32_MAX;
                for (int v = 0; v < N; v++)
                {
                    int residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
                    if (residual_cap > 0)
                    {
                        min_dist = min(min_dist, dist[v]);
                        stash_dist[u] = min_dist + 1;
                    }
                }
            }
        }
        print_array(stash_dist, N);
        // Stage 3: update modified dist.
        sync_dist(dist, stash_dist, N, processRank, nProcess, globalActiveNodes, CastvCnt, CastvDisp, comm);
        print_array(dist, N);

        for (int v = 0; v < N; v++)
        {
            if (stash_excess[v] != 0)
            {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // stage 4: rebuild active nodes
        globalActiveNodes.clear();
        for (auto u=0;u<N;u++){
            if (excess[u] > 0 && u != src && u != sink) {
                globalActiveNodes.emplace_back(u);
            }
        }
        nGlobalActiveNodes = globalActiveNodes.size();
#ifdef DEBUG    
        if(processRank == 0){
            printf("nGlobalActiveNodes=%d\n", nGlobalActiveNodes);
            printf("=========================================\n");
        }
#endif
    }
    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    if(processRank != 0){
        free(flow);
        free(cap);
    }

    return 0;
}
