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
#include <chrono>
#include "unistd.h"
#include "mpi_push_relabel.h"

#define DEBUG
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

long long int send_time[4] = {0};

void print_send_time(int processRank) {
    for(int i=0;i<1;i++){
        printf("[%d]send %d time: %lld us\n", processRank, i, send_time[i]);
    }
}

int send_inverse_flow(int outProcIdx, vector<vector<InverseFlowPair>> &outInverseFlow, MPI_Comm comm){
    auto start_clock = high_resolution_clock::now();
    int nOutInverseFlow;
    nOutInverseFlow = outInverseFlow[outProcIdx].size();
    MPI_Send(&nOutInverseFlow, 1, MPI_INT, outProcIdx, N_INVERSE_FLOW_TAG, comm);
    if(nOutInverseFlow > 0){
        MPI_Send(outInverseFlow[outProcIdx].data(), nOutInverseFlow, MpiInverseFlowPair, outProcIdx, INVERSE_FLOW_DATA_TAG, comm);
    }
    send_time[0] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
    return 0;
}

long long int recv_time[4] = {0};

void print_recv_time(int processRank) {
    for(int i=0;i<3;i++){
        printf("[%d]recv %d time: %lld us\n", processRank, i, recv_time[i]);
    }
}

vector<InverseFlowPair> recvBuff;
int recv_inverse_flow(unordered_set<int> &waitProcesses, vector<InverseFlowPair> &recvInverseFlowPair, MPI_Comm comm, int processRank){
    // printf("][");
    int nWaitRecv = waitProcesses.size();
    unordered_map<int, int> nInInverseFlows;
    while(nWaitRecv > 0){
        auto start_clock = high_resolution_clock::now();
        MPI_Status probeStatus;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &probeStatus);
        long long int probeTime = duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
        // printf("%d:%lld ", processRank, probeTime);
        recv_time[0] += probeTime;
        start_clock = high_resolution_clock::now();
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
                MPI_Recv(recvBuff.data(), recvSize, MpiInverseFlowPair, source, tag, comm, &recvStatus);
                for(int i=0; i<recvSize; i++){
                    recvInverseFlowPair.push_back(recvBuff[i]);
                }
                nWaitRecv--;
            }
        }
        recv_time[1] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
    }
    return 0;
}

int sync_flow(int* flow, int N, int processRank, int nProcess, MPI_Comm comm, vector<vector<InverseFlowPair>> &outInverseFlow){
    // 发送outInverseFlow并更新各自flow
    vector<InverseFlowPair> recvInverseFlowPair;
    // phase 1: 发给大的，收小的
    for(int outProcIdx = outInverseFlow.size()-1; outProcIdx > processRank; outProcIdx--){
        send_inverse_flow(outProcIdx, outInverseFlow, comm);
    }

    unordered_set<int> waitProcesses;
    for(int i=0;i<processRank;i++){
        waitProcesses.insert(i);
    }
    recv_inverse_flow(waitProcesses, recvInverseFlowPair, comm, processRank);
    MPI_Barrier(comm);

    // phase 2: 发给小的，收大的
    for(int outProcIdx = 0; outProcIdx < processRank; outProcIdx++){
        send_inverse_flow(outProcIdx, outInverseFlow, comm);
    }

    waitProcesses.clear();
    for(int i=nProcess-1;i>processRank;i--){
        waitProcesses.insert(i);
    }
    recv_inverse_flow(waitProcesses, recvInverseFlowPair, comm, processRank);

    // phase 3: 自己
    for(auto ptr = outInverseFlow[processRank].begin(); ptr != outInverseFlow[processRank].end(); ptr++){
        recvInverseFlowPair.push_back(*ptr);
    }
    // phase 4: update flow
    for(auto ptr = recvInverseFlowPair.begin(); ptr != recvInverseFlowPair.end(); ptr++){
        int src = ptr->src, dst = ptr->dst, val = ptr->val;
        flow[utils::idx(src, dst, N)] -= val;
    }

    // phase 1: send/recv number
    // vector<int> outSize;
    // for(int i=0;i<outInverseFlow.size();i++){
    //     outSize.emplace_back(outInverseFlow[i].size());
    // }

    // vector<int> inSize;
    // inSize.resize(nProcess*nProcess);
    // auto start_clock = high_resolution_clock::now();
    // MPI_Allgather(outSize.data(), nProcess, MPI_INT, inSize.data(), nProcess, MPI_INT, comm);

    // recv_time[0] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
    // start_clock = high_resolution_clock::now();
    // // phase 2: send/recv data
    // vector<MPI_Request> dataRequests;
    // dataRequests.resize(2*(nProcess-1));
    // int reqIdx = 0;
    // vector<vector<InverseFlowPair>> recvPairs;
    // recvPairs.resize(nProcess);
    // for(int i=0;i<nProcess;i++){
    //     if(i != processRank){
    //         if(outSize[i] > 0){
    //             MPI_Isend(outInverseFlow[i].data(), outSize[i], MpiInverseFlowPair, i, INVERSE_FLOW_DATA_TAG, comm, &dataRequests[reqIdx]);
    //             reqIdx++;
    //         }
    //         if(inSize[i*nProcess + processRank] > 0){
    //             recvPairs[i].resize(inSize[i*nProcess + processRank]);
    //             MPI_Irecv(recvPairs[i].data(), inSize[i*nProcess + processRank], MpiInverseFlowPair, i, INVERSE_FLOW_DATA_TAG, comm, &dataRequests[reqIdx]);
    //             reqIdx++;
    //         }
    //     }
    // }
    // MPI_Waitall(reqIdx, dataRequests.data(), MPI_STATUSES_IGNORE);
    // recv_time[1] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
    
    // start_clock = high_resolution_clock::now();
    // // phase 3: update flow
    // for(auto ptr = outInverseFlow[processRank].begin(); ptr != outInverseFlow[processRank].end(); ptr++){
    //     int src = ptr->src, dst = ptr->dst, val = ptr->val;
    //     flow[utils::idx(src, dst, N)] -= val;
    // }
    // for(int i=0;i<nProcess;i++){
    //     if(i != processRank && inSize[i] > 0){
    //         for(auto ptr = recvPairs[i].begin(); ptr != recvPairs[i].end(); ptr++){
    //             int src = ptr->src, dst = ptr->dst, val = ptr->val;
    //             flow[utils::idx(src, dst, N)] -= val;
    //         }
    //     }
    // }
    // recv_time[2] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();


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
    get_castv_cnt_disp_array(N, nProcess, CastvCnt, CastvDisp);

    // alloc mem for immediate vars
    int *dist = (int *)calloc(N, sizeof(int));
    int *stash_dist = (int *)calloc(N, sizeof(int));
    int64_t *excess = (int64_t *)calloc(N, sizeof(int64_t));
    int64_t *stash_excess = (int64_t *)calloc(N, sizeof(int64_t));
    int *stash_send = (int *)calloc(N * N, sizeof(int));
    int64_t *new_stash_excess = (int64_t *)calloc(N, sizeof(int64_t));
    recvBuff.resize(N*N);


    // do first flow
    pre_flow(dist, excess, cap, flow, N, src);

    // compute workload (number of vertex managed by this process)
    int nWorkload = get_workload(processRank, N, nProcess);

    vector<int> globalActiveNodes;
    vector<int> localActiveNodes;
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

    long long int timer[5] = {0};
    // make data ready use about 0.01s on dataset2, so it should not be the reason of slow
    // Four-Stage Pulses.
    while (nGlobalActiveNodes > 0) {
        // clear immediate vars
        auto start_clock = high_resolution_clock::now();
        memset(stash_excess, 0, sizeof(int64_t)*N);
        memset(new_stash_excess, 0, sizeof(int64_t)*N);

        /*
        Stage 1:
        each process of u should be parallelized.
        flow need to be broadcast, each process maintain their stash_send
        each process modify their own stash_send and excess
        */

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
        Stage 2:
        each process of u should be parallelized
        each worker works on its rows in flow, and contributes to stash_excess at each v
        after all master reduce all stash_excess
        */

        /* serial
        for (int u : localActiveNodes)
        {
            for (int v = 0; v < N; v++)
            {
                if (stash_send[utils::idx(u, v, N)] > 0)
                {
                    // 这里如果按u划分，那么stash_send不必传送，如果按v划分，则stash_excess不必传送。考虑stash_send规模更大， 采用按u划分。
                    flow[utils::idx(u, v, N)] += stash_send[utils::idx(u, v, N)];
                    flow[utils::idx(v, u, N)] -= stash_send[utils::idx(u, v, N)];
                    stash_excess[v] += stash_send[utils::idx(u, v, N)];
                    stash_send[utils::idx(u, v, N)] = 0;
                }
            }
        }
        */

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
        timer[0] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();

        start_clock = high_resolution_clock::now();

        MPI_Barrier(comm);
        timer[1] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();

        // sync flow data across processes
        sync_flow(flow, N, processRank, nProcess, comm, outInverseFlow);
        print_array(flow, N*N);


        // sum up all stash_excess
        MPI_Request excessRequest;
        // MPI_Allreduce(stash_excess, new_stash_excess, N, MPI_INT64_T, MPI_SUM, comm);
        MPI_Iallreduce(stash_excess, new_stash_excess, N, MPI_INT64_T, MPI_SUM, comm, &excessRequest);

        start_clock = high_resolution_clock::now();

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        /*
        Stage 3:
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
        timer[2] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();
        start_clock = high_resolution_clock::now();

        timer[3] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();

        // update dist async.
        MPI_Request distRequest;
        MPI_Iallgatherv(&stash_dist[CastvDisp[processRank]], CastvCnt[processRank], MPI_INT, dist, CastvCnt.data(), CastvDisp.data(), MPI_INT, comm, &distRequest);
        // MPI_Allgatherv(&stash_dist[CastvDisp[processRank]], CastvCnt[processRank], MPI_INT, dist, CastvCnt.data(), CastvDisp.data(), MPI_INT, comm);
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
        start_clock = high_resolution_clock::now();

        // check finish of stash_excess
        MPI_Wait(&excessRequest, MPI_STATUS_IGNORE);
        memcpy(stash_excess, new_stash_excess, sizeof(int64_t)*N);
        timer[4] += duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count();


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

        // wait update of dist
        MPI_Wait(&distRequest, MPI_STATUS_IGNORE);


#ifdef DEBUG    
        if(processRank == 0){
            printf("nGlobalActiveNodes=%d\n", nGlobalActiveNodes);
            printf("=========================================\n");
        }
#endif
    }
    for(int i=0;i<5;i++){
        printf("[p%d]stage %d consume %lld us.\n", processRank, i, timer[i]);
    }
    // 更新process 0 上的flow
    vector<int> flowCastvDisp(CastvDisp), flowCastvCnt(CastvCnt);
    for(int i=0;i<nProcess;i++) {
        flowCastvDisp[i] *= N;
        flowCastvCnt[i] *= N;
    }

    auto start_clock = high_resolution_clock::now();
    MPI_Gatherv(&flow[flowCastvDisp[processRank]], flowCastvCnt[processRank], MPI_INT, 
    flow, flowCastvCnt.data(), flowCastvDisp.data(), MPI_INT, 0, comm);
    printf("gather flow consume %lld us\n", duration_cast<microseconds>(high_resolution_clock::now() - start_clock).count());

    print_recv_time(processRank);
    print_send_time(processRank);
    free(dist);
    free(stash_dist);
    free(excess);
    free(stash_excess);
    free(stash_send);
    free(new_stash_excess);

    if(processRank != 0){
        free(flow);
        free(cap);
    }

    return 0;
}
