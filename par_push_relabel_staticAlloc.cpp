/*
本实现中固定的vertex分配给固定的process，这样就不需要每次循环都全局更新flow。
但是有可能会导致后期的负载不均衡问题。
*/

#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include "unistd.h"
#include "mpi_push_relabel.h"

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

typedef struct InverseFlowPair{
    int src;
    int dst;
    int val;
}InverseFlowPair;

MPI_Datatype MpiInverseFlowPair;

// template<class T>
// void print_array(T *array, int length, int processId){
//     cout << "[proc" << processId << "]:";
//     cout << "{";
//     for(int i=0;i<length;i++){
//         cout << array[i] << ", ";
//     }
//     cout << "}" << endl;
// }

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

/*
input vertexId, N and nWorkers
output processid that process this vertex and offset in its node array
*/
int get_process_vid_offset(int N, int nWorkers, vector<int> &vid2process, vector<int> &vid2offset){
    if(N%nWorkers == 0){
        int nBaseWorkload = N/nWorkers;
        for(int i=0;i<nWorkers;i++){
            for(int j=0;j<nBaseWorkload;j++){
                vid2process.push_back(i);
                vid2offset.push_back(j);
            }
        }
    } else {
        int nBaseWorkload = N/nWorkers;
        int nHeavyProc = N%nWorkers;
        for(int i=0;i<nWorkers;i++){
            int nWorkload = i<nHeavyProc?nBaseWorkload+1:nBaseWorkload;
            for(int j=0;j<nWorkload;j++){
                vid2process.push_back(i);
                vid2offset.push_back(j);
            }
        }
    }
    return 0;
}

int get_workload(int processRank, int N, int nWorkers){
    if(N%nWorkers == 0){
        return N/nWorkers;
    } else {
        int nBaseWorkload = N/nWorkers;
        int nHeavyProc = N%nWorkers;
        if (processRank < nHeavyProc) {
            return nBaseWorkload+1;
        } else {
            return nBaseWorkload;
        }
    }
}

int get_castv_cnt_disp_array(int N, int nWorkers, vector<int> &cnt, vector<int> &disp){
    cnt.clear(); disp.clear();
    int nBaseWorkload = N/nWorkers;
    int nHeavyProc = N%nWorkers;
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

/*
exchange inverse flow helper function
*/
#define N_INVERSE_FLOW_TAG 1
#define INVERSE_FLOW_DATA_TAG 2

int send_inverse_flow(int outProcIdx, vector<vector<InverseFlowPair>> &outInverseFlow, MPI_Comm comm){
    int nOutInverseFlow;
    nOutInverseFlow = outInverseFlow[outProcIdx].size();
    MPI_Send(&nOutInverseFlow, 1, MPI_INT, outProcIdx, N_INVERSE_FLOW_TAG, comm);
    if(nOutInverseFlow > 0){
        MPI_Send(outInverseFlow[outProcIdx].data(), nOutInverseFlow, MpiInverseFlowPair, outProcIdx, INVERSE_FLOW_DATA_TAG, comm);
    }
    return 0;
}

int recv_inverse_flow(unordered_set<int> &waitProcesses, vector<InverseFlowPair> &recvInverseFlowPair, MPI_Comm comm){
    int nWaitRecv = waitProcesses.size();
    unordered_map<int, int> nInInverseFlows;
    while(nWaitRecv > 0){
        MPI_Status probeStatus;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &probeStatus);
        if(probeStatus.MPI_TAG == N_INVERSE_FLOW_TAG){
            MPI_Status recvStatus;
            int source = probeStatus.MPI_SOURCE;
            int tag = probeStatus.MPI_TAG;
            int sizeInInverseFlows = -1;
            MPI_Recv(&sizeInInverseFlows, 1, MPI_INT, source, tag, comm, &recvStatus);
            nInInverseFlows[source] = sizeInInverseFlows;
            if(sizeInInverseFlows == 0){
                nWaitRecv--;
            }
        } else if (probeStatus.MPI_TAG == INVERSE_FLOW_DATA_TAG){
            int source = probeStatus.MPI_SOURCE;
            int tag = probeStatus.MPI_TAG;
            if(nInInverseFlows[source] == -1){
                continue; // 暂时忽略，因为还没收到有多少条。如果MPI是TCP实现，应该不会出现这种情况。
            } else {
                MPI_Status recvStatus;
                int recvSize = nInInverseFlows[source];
                vector<InverseFlowPair> recvBuff;
                recvBuff.resize(recvSize, InverseFlowPair{0,0,0});
                MPI_Recv(recvBuff.data(), recvSize, MpiInverseFlowPair, source, tag, comm, &recvStatus);
                for(auto p = recvBuff.begin(); p!= recvBuff.end(); p++){
                    recvInverseFlowPair.push_back(*p);
                }
                nWaitRecv--;

            }
        }
    }
    return 0;
}

int sync_flow(int* flow, int N, int processRank, int nProcess, MPI_Comm comm, vector<vector<InverseFlowPair>> &outInverseFlow){
    // 发送outInverseFlow并更新各自flow
    vector<InverseFlowPair> recvInverseFlowPair;
    // phase 1: 发给大的，收小的
    // send phase
    for(int outProcIdx = outInverseFlow.size()-1; outProcIdx > processRank; outProcIdx--){
        send_inverse_flow(outProcIdx, outInverseFlow, comm);
    }
    // recv phase
    unordered_set<int> waitProcesses;
    for(int i=0;i<processRank;i++){
        waitProcesses.insert(i);
    }
    recv_inverse_flow(waitProcesses, recvInverseFlowPair, comm);
    // barrier
    // MPI_Barrier(comm);

    // phase 2: 发给小的，收大的
    // send phase
    for(int outProcIdx = 0; outProcIdx < processRank; outProcIdx++){
        send_inverse_flow(outProcIdx, outInverseFlow, comm);
    }
    // recv phase
    waitProcesses.clear();
    for(int i=nProcess-1;i>processRank;i--){
        waitProcesses.insert(i);
    }
    recv_inverse_flow(waitProcesses, recvInverseFlowPair, comm);

    // phase 3: 自己
    for(auto ptr = outInverseFlow[processRank].begin(); ptr != outInverseFlow[processRank].end(); ptr++){
        recvInverseFlowPair.push_back(*ptr);
    }
    // phase 4: update flow
    for(auto ptr = recvInverseFlowPair.begin(); ptr != recvInverseFlowPair.end(); ptr++){
        int src = ptr->src, dst = ptr->dst, val = ptr->val;
        flow[utils::idx(src, dst, N)] -= val;
    }
    return 0;
}

int push_relabel(int my_rank, int p, MPI_Comm comm, int N, int src, int sink, int *cap, int *flow) {
    register_inverse_flow_pair_to_mpi(&MpiInverseFlowPair);
    int nProcess=p, processRank=my_rank;
    /*
    ready:
    PStage 1:
    cap should be broadcast to each process
    */
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);
    MPI_Bcast(&src, 1, MPI_INT, 0, comm);
    MPI_Bcast(&sink, 1, MPI_INT, 0, comm);
    if (processRank != 0) {
        cap = (int *)calloc(N * N, sizeof(int)); // todo: remember to free!
        flow = (int *)calloc(N * N, sizeof(int)); // todo: remember to free!
    }
    MPI_Bcast(cap, N * N, MPI_INT, 0, comm);
    vector<int> CastvCnt, CastvDisp;
    get_castv_cnt_disp_array(N, nProcess, CastvCnt, CastvDisp);

    /*
    parallel in a master-worker paradism
    process 0 acts as the master, and all processes act as workers
    */
    int *dist = (int *)calloc(N, sizeof(int));
    int *stash_dist = (int *)calloc(N, sizeof(int));
    int64_t *excess = (int64_t *)calloc(N, sizeof(int64_t));
    int64_t *stash_excess = (int64_t *)calloc(N, sizeof(int64_t));

    /*
    PreFlow:
    PStage 2:
    no need to parallel, high communication compared to computation
    */
    pre_flow(dist, excess, cap, flow, N, src);

    vector<int> globalActiveNodes;
    vector<int> localActiveNodes;
    int nWorkload = get_workload(processRank, N, nProcess);

    int *stash_send = (int *)calloc(N * N, sizeof(int));

    vector<int> vid2process, vid2offset;
    get_process_vid_offset(N, nProcess, vid2process, vid2offset);
    /*
    PStage 3:
    active nodes are managed by process 0
    */
    for (int u = 0; u < N; u++) {
        if (u != src && u != sink) {
            globalActiveNodes.emplace_back(u);
            if(vid2process[u] == processRank){
                localActiveNodes.emplace_back(u);
            }
        }
    }
    int nGlobalActiveNodes = globalActiveNodes.size();
    print_array(cap, N*N);
    // make data ready use about 0.01s on dataset2, so it should not be the reason of slow
    // Four-Stage Pulses.
    int nloop = 0;
    while (nGlobalActiveNodes > 0) {
        nloop++;
        // Stage 1: push.
        // split tasks, use block partition strategy (3 3 2 2 2)
        memset(stash_excess, 0, sizeof(int64_t)*N);

        /*
        PStage 4:
        each process of u should be parallelized.
        flow need to be broadcast, each process maintain their stash_send
        each process modify their own stash_send and excess
        */

        /* same as serial Stage 1*/
        print_array(dist, N);
        for (int u : localActiveNodes)
        {
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
        print_array(stash_send, N*N);
        print_array(excess, N);

        /*
        PStage 5:
        each process of u should be parallelized
        each worker works on its rows in flow, and contributes to stash_excess at each v
        after all master reduce all stash_excess
        */

        // for (int u : localActiveNodes)
        // {
        //     for (int v = 0; v < N; v++)
        //     {
        //         if (stash_send[utils::idx(u, v, N)] > 0)
        //         {
        //             // 这里如果按u划分，那么stash_send不必传送，如果按v划分，则stash_excess不必传送。考虑stash_send规模更大， 采用按u划分。
        //             flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
        //             flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
        //             stash_excess[v] += stash_send[utils::idx(u, v, N)];
        //             stash_send[utils::idx(u, v, N)] = 0;
        //         }
        //     }
        // }

        vector<vector<InverseFlowPair>> outInverseFlow;
        outInverseFlow.resize(nProcess, vector<InverseFlowPair>());
        for (int u : localActiveNodes)
        {
            for (int v = 0; v < N; v++)
            {
                if (stash_send[utils::idx(u, v, N)] > 0)
                {
                    // 这里如果按u划分，那么stash_send不必传送，如果按v划分，则stash_excess不必传送。考虑stash_send规模更大， 采用按u划分。
                    flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    InverseFlowPair inverseFlow;
                    inverseFlow.dst = u;
                    inverseFlow.src = v;
                    inverseFlow.val = stash_send[utils::idx(u, v, N)];
                    int outProcessIdx = vid2process[v];
                    outInverseFlow[outProcessIdx].push_back(inverseFlow);
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }
        pause;
        // MPI_Barrier(comm);
        sync_flow(flow, N, processRank, nProcess, comm, outInverseFlow);

        // MPI_Barrier(comm);
        // 收集stash_excess并发送给所有人
        int64_t *new_stash_excess = (int64_t *)calloc(N, sizeof(int64_t));
        MPI_Allreduce(stash_excess, new_stash_excess, N, MPI_INT64_T, MPI_SUM, comm);
        memcpy(stash_excess, new_stash_excess, sizeof(int64_t)*N);
        free(new_stash_excess);

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        /*
        PStage 6:
        each process of u are parallelized
        master send corresponding excess and flow to workers
        master send all dist to workers, workers generate stash_dist
        workers update their elements in stash_dist and send back to master, master update them to dist
        */
        for (int u : localActiveNodes)
        {
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

        // Stage 3: update dist.
        MPI_Allgatherv(&stash_dist[CastvDisp[processRank]], CastvCnt[processRank], MPI_INT, dist, CastvCnt.data(), CastvDisp.data(), MPI_INT, comm);
        // swap(dist, stash_dist);

        // barrier
        /*
        PStage 7:
        此时stash_excess是全部正确的，但excess仅自己管理的部分是正确的
        因此只更新自己管理的部分
        */
        // Stage 4: apply excess-flow changes for destination vertices.
        /*
        // [serial version]
        for (int v = 0; v < N; v++)
        {
            if (stash_excess[v] != 0)
            {
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }
        */
        for(int i=0;i<CastvCnt[processRank];i++){
            int v = i + CastvDisp[processRank];
            if(stash_excess[v] != 0){
                excess[v] += stash_excess[v];
                stash_excess[v] = 0;
            }
        }

        // Construct active nodes.
        // by master, no need to parallel
        /*
        // [serial version]
        active_nodes.clear();
        for (auto u = 0; u < N; u++) {
            if (excess[u] > 0 && u != src && u != sink) {
                active_nodes.emplace_back(u);
            }
        }
        */
        localActiveNodes.clear();
        for(int i=0;i<CastvCnt[processRank];i++){
            int v = i + CastvDisp[processRank];
            if(excess[v] > 0 && v != src && v != sink) {
                localActiveNodes.emplace_back(v);
            }
        }
        int nLocalActiveNodes = localActiveNodes.size();
        MPI_Allreduce(&nLocalActiveNodes, &nGlobalActiveNodes, 1, MPI_INT, MPI_SUM, comm);
#ifdef DEBUG    
        if(processRank == 0){
            printf("nGlobalActiveNodes=%d\n", nGlobalActiveNodes);
            printf("=========================================\n");
        }
#endif
        // MPI_Barrier(MPI_COMM_WORLD);
    }
    printf("total loop: %d", nloop);
    clock_t t1 = clock();
    // 更新process 0 上的flow
    vector<int> flowCastvDisp(CastvDisp), flowCastvCnt(CastvCnt);
    for(int i=0;i<nProcess;i++){
        flowCastvDisp[i] *= N;
        flowCastvCnt[i] *= N;
    }
    MPI_Gatherv(&flow[flowCastvDisp[processRank]], flowCastvCnt[processRank], MPI_INT, 
    flow, flowCastvCnt.data(), flowCastvDisp.data(), MPI_INT, 0, comm);
    clock_t t2 = clock();
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
