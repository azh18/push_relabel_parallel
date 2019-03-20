/**
 * Name:
 * Student id:
 * ITSC email:
*/


#include "pthread_push_relabel.h"
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <vector>
#include <iostream>
#include <chrono>
#include <pthread.h>

using namespace std;
using namespace std::chrono;

#ifdef DEBUG
#define print_array(x, l) cout << #x << ":{"; \
    for(int i=0;i<l;i++){ \
        cout << x[i] << ", "; \
    } \
    cout << "}" << endl;
#define pause sleep(1);
#else
#define print_array(x, l) 1;
#define pause sleep(0);
#endif

#ifdef __APPLE__
typedef int pthread_barrierattr_t;
typedef struct
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int tripCount;
} pthread_barrier_t;


int pthread_barrier_init(pthread_barrier_t *barrier, const pthread_barrierattr_t *attr, unsigned int count)
{
    if(count == 0)
    {
        errno = EINVAL;
        return -1;
    }
    if(pthread_mutex_init(&barrier->mutex, 0) < 0)
    {
        return -1;
    }
    if(pthread_cond_init(&barrier->cond, 0) < 0)
    {
        pthread_mutex_destroy(&barrier->mutex);
        return -1;
    }
    barrier->tripCount = count;
    barrier->count = 0;

    return 0;
}

int pthread_barrier_destroy(pthread_barrier_t *barrier)
{
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t *barrier)
{
    pthread_mutex_lock(&barrier->mutex);
    ++(barrier->count);
    if(barrier->count >= barrier->tripCount)
    {
        barrier->count = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1;
    }
    else
    {
        pthread_cond_wait(&barrier->cond, &(barrier->mutex));
        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}
#endif // __APPLE__

typedef struct InverseFlow{
    int u;
    int v;
    int val;
}InverseFlow;

typedef struct Stage1Args{
    int N;
    int tid;
    int* cap;
    int* flow;
    int* dist;
    int64_t* excess;
    int64_t* stash_excess;
    int64_t* stash_stash_excess;
    vector<int> *active_nodes;
    vector<int> *disp;
    vector<int> *cnt;
    vector<InverseFlow> *inverseFlows;
}Stage1Args;

typedef struct Stage2Args{
    int N;
    int tid;
    int* cap;
    int* flow;
    int* dist;
    int *stash_dist;
    int64_t* excess;
    vector<int> *active_nodes;
    vector<int> *disp;
    vector<int> *cnt;
}Stage2Args;


pthread_mutex_t stash_excess_mutex, flow_mutex;
pthread_barrier_t flow_write_barrier;

void* do_stage_1(void* args){
    Stage1Args* pArgs = (Stage1Args*)args;
    int N = pArgs->N;
    int tid = pArgs->tid;
    int* cap = pArgs->cap;
    int* flow = pArgs->flow;
    int* dist = pArgs->dist;
    int64_t* excess = pArgs->excess;
    int64_t* stash_excess = pArgs->stash_excess;
    int64_t* stash_stash_excess = pArgs->stash_stash_excess;
    vector<int> &active_nodes = *(pArgs->active_nodes);
    vector<int> &disp = *(pArgs->disp);
    vector<int> &cnt = *(pArgs->cnt);
    vector<InverseFlow> &inverseFlows = *(pArgs->inverseFlows);
    inverseFlows.clear();

    int start = disp[tid];
    int end = disp[tid]+cnt[tid];
    for(int idx=start; idx<end; idx++){
        int u = active_nodes[idx];
        for(auto v = 0; v< N; v++){
            auto residual_cap = cap[utils::idx(u, v, N)] -
                flow[utils::idx(u, v, N)];
            if (residual_cap > 0 && dist[u] > dist[v] && excess[u] > 0) {
                int send_u_to_v = std::min<int64_t>(excess[u], residual_cap); // no need to justify send_u_to_v, because it must > 0
                excess[u] -= send_u_to_v;
                inverseFlows.emplace_back(InverseFlow{u,v,send_u_to_v});
                stash_stash_excess[v] += send_u_to_v;
            }
        }
    }

    pthread_mutex_lock(&stash_excess_mutex);
    for(int v=0;v<N;v++){
        stash_excess[v] += stash_stash_excess[v];
    }
    pthread_mutex_unlock(&stash_excess_mutex);
    pthread_barrier_wait(&flow_write_barrier);
    pthread_mutex_lock(&flow_mutex);
    for(auto flow_unit: inverseFlows){
        int u = flow_unit.u;
        int v = flow_unit.v;
        int val = flow_unit.val;
        flow[utils::idx(u, v, N)] += val;
        flow[utils::idx(v, u, N)] -= val;
    }
    pthread_mutex_unlock(&flow_mutex);
    return NULL;
}

void* do_stage_2(void* origin_args) {
    Stage2Args* args = (Stage2Args*) origin_args;
    int N = args->N;
    int tid = args->tid;
    int *cap = args->cap, *flow = args->flow, *dist = args->dist, *stash_dist = args->stash_dist;
    int64_t *excess = args->excess;
    vector<int> &active_nodes = *(args->active_nodes);
    vector<int> &disp = *(args->disp);
    vector<int> &cnt = *(args->cnt);
    int start = disp[tid];
    int end = disp[tid]+cnt[tid];
    for(int idx=start; idx<end; idx++){
        int u = active_nodes[idx];
        int min_dist = INT32_MAX;
        for (auto v = 0; v < N; v++) {
            auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
            if (residual_cap > 0) {
                min_dist = min(min_dist, dist[v]);
                stash_dist[u] = min_dist + 1;
            }
        }
    }
    return NULL;
}

/*
    thread pool
*/

typedef struct TaskNotify{
    pthread_cond_t inCond;
    pthread_mutex_t inMutex;
}TaskNotify;

typedef struct FinishNotify{
    pthread_cond_t cond;
    pthread_mutex_t mutex;
    int nTotalThreads;
    int nFinished;
}FinishNotify;

class Thread{
    public:
        static vector<Thread> thread_pool;
        static void createThreadPool(int n);
        static void cleanThreadPool();
        static FinishNotify finish_notify;
        static void init_finish_notify();
        static void* thread_handler(void* thread_obj);
        static void run_stage_1_task(vector<Stage1Args>& args);
        static void run_stage_2_task(vector<Stage2Args>& args);
        pthread_t thread;
        int now_task_id;
        Stage1Args *args1;
        Stage2Args *args2;
        TaskNotify notify;
        Thread();
        ~Thread();
        void* handler();
        void add_stage_1_task(Stage1Args* args);
        void add_stage_2_task(Stage2Args* args);
};

vector<Thread> Thread::thread_pool;
FinishNotify Thread::finish_notify;


Thread::Thread(){
    pthread_mutex_init(&this->notify.inMutex, NULL);
    pthread_cond_init(&this->notify.inCond, NULL);
    this->args1 = NULL;
    this->args2 = NULL;
    this->now_task_id = 0;
    pthread_create(&this->thread, NULL, Thread::thread_handler, (void*)this);
    pthread_detach(this->thread);
}

Thread::~Thread(){
    pthread_mutex_destroy(&this->notify.inMutex);
    pthread_cond_destroy(&this->notify.inCond);
}

void Thread::add_stage_1_task(Stage1Args* args){
    pthread_mutex_lock(&this->notify.inMutex);
    this->now_task_id = 1;
    this->args1 = args;
    this->args2 = NULL;
    pthread_cond_signal(&this->notify.inCond);
    pthread_mutex_unlock(&this->notify.inMutex);
}

void Thread::add_stage_2_task(Stage2Args* args){
    pthread_mutex_lock(&this->notify.inMutex);
    this->now_task_id = 2;
    this->args1 = NULL;
    this->args2 = args;
    pthread_cond_signal(&this->notify.inCond);
    pthread_mutex_unlock(&this->notify.inMutex);
}

void* Thread::thread_handler(void* thread_obj){
    Thread* obj = (Thread*)thread_obj;
    obj->handler();
    return NULL;
}

void Thread::run_stage_1_task(vector<Stage1Args>& args){
    pthread_mutex_lock(&Thread::finish_notify.mutex);
    Thread::finish_notify.nFinished = 0;
    Thread::finish_notify.nTotalThreads = Thread::thread_pool.size();
    for(int i=0;i<Thread::thread_pool.size();i++){
        Thread::thread_pool[i].add_stage_1_task(&args[i]);
    }
    pthread_cond_wait(&Thread::finish_notify.cond, &Thread::finish_notify.mutex);
    pthread_mutex_unlock(&Thread::finish_notify.mutex);
}

void Thread::run_stage_2_task(vector<Stage2Args>& args){
    pthread_mutex_lock(&Thread::finish_notify.mutex);
    Thread::finish_notify.nFinished = 0;
    Thread::finish_notify.nTotalThreads = Thread::thread_pool.size();
    for(int i=0;i<Thread::thread_pool.size();i++){
        Thread::thread_pool[i].add_stage_2_task(&args[i]);
    }
    pthread_cond_wait(&Thread::finish_notify.cond, &Thread::finish_notify.mutex);
    pthread_mutex_unlock(&Thread::finish_notify.mutex);
}

void Thread::init_finish_notify(){
    pthread_mutex_init(&Thread::finish_notify.mutex, NULL);
    pthread_cond_init(&Thread::finish_notify.cond, NULL);
    Thread::finish_notify.nFinished = 0;
}

void Thread::createThreadPool(int n){
    Thread::init_finish_notify();
    Thread::finish_notify.nTotalThreads = n;
    Thread::thread_pool.resize(n);
}

void Thread::cleanThreadPool(){
    int nThread = Thread::thread_pool.size();
    for(int i=0;i<nThread;i++){
        pthread_mutex_lock(&Thread::thread_pool[i].notify.inMutex);
        Thread::thread_pool[i].now_task_id = -1;
        pthread_cond_signal(&Thread::thread_pool[i].notify.inCond);
        pthread_mutex_unlock(&Thread::thread_pool[i].notify.inMutex);
    }
    pthread_mutex_destroy(&Thread::finish_notify.mutex);
    pthread_cond_destroy(&Thread::finish_notify.cond);
}

void* Thread::handler(){
    while(1){
        pthread_mutex_lock(&notify.inMutex);
        if(this->now_task_id == 0)
            pthread_cond_wait(&notify.inCond, &notify.inMutex);
        pthread_mutex_unlock(&notify.inMutex);
        // finish task
        switch (this->now_task_id)
        {
            case 1:
                do_stage_1(this->args1);
                break;
            case 2:
                do_stage_2(this->args2);
                break;
            case -1:
                return NULL;
            default:
                printf("Warn: Task undefined!\n");
                break;
        }
        this->now_task_id = 0;
        pthread_mutex_lock(&Thread::finish_notify.mutex);
        Thread::finish_notify.nFinished++;
        if(Thread::finish_notify.nFinished >= Thread::finish_notify.nTotalThreads){
            pthread_cond_signal(&Thread::finish_notify.cond);
        }
        pthread_mutex_unlock(&Thread::finish_notify.mutex);
    }
    return NULL;
}

/*
 *  You can add helper functions and variables as you wish.
 */

void pre_flow(int *dist, int64_t *excess, int *cap, int *flow, int N, int src) {
    dist[src] = N;
    for (auto v = 0; v < N; v++) {
        flow[utils::idx(src, v, N)] = cap[utils::idx(src, v, N)];
        flow[utils::idx(v, src, N)] = -flow[utils::idx(src, v, N)];
        excess[v] = flow[utils::idx(src, v, N)];
    }
}

void get_divide(int num_threads, int tasks, vector<int> &disp, vector<int> &cnt){
    disp.clear();
    cnt.clear();
    int n_large_threads = tasks % num_threads;
    int n_tasks_per_thread = tasks / num_threads;
    int offset = 0;
    for(int i=0;i<num_threads;i++){
        disp.emplace_back(offset);
        if(i < n_large_threads){
            cnt.emplace_back(n_tasks_per_thread + 1);
            offset += (n_tasks_per_thread + 1);
        } else{
            cnt.emplace_back(n_tasks_per_thread);
            offset += n_tasks_per_thread;
        }
    }
    return;
}

int push_relabel(int num_threads, int N, int src, int sink, int *cap, int *flow) {
    /*
     *  Please fill in your codes here.
     */
    int *dist = (int *) calloc(N, sizeof(int));
    int *stash_dist = (int *) calloc(N, sizeof(int));
    auto *excess = (int64_t *) calloc(N, sizeof(int64_t));
    auto *stash_excess = (int64_t *) calloc(N, sizeof(int64_t));

    /*
        memory seperated by threads
    */
    int64_t* stash_stash_excess = (int64_t *) calloc(N*num_threads, sizeof(int64_t));

    // PreFlow.
    pre_flow(dist, excess, cap, flow, N, src);

    // task divide
    Thread::createThreadPool(num_threads);

    vector<int> disp, cnt;
    vector<int> active_nodes;
    vector<int> active_nodes_with_excess;
    int *stash_send = (int *) calloc(N * N, sizeof(int));

    for (auto u = 0; u < N; u++) {
        if (u != src && u != sink) {
            active_nodes.emplace_back(u);
        }
    }

    pthread_mutex_init(&stash_excess_mutex, NULL);
    pthread_mutex_init(&flow_mutex, NULL);
    vector<Stage1Args> args_pool_stage1;
    vector<Stage2Args> args_pool_stage2;
    vector<vector<InverseFlow> > inverse_flow_pool;
    inverse_flow_pool.resize(num_threads);

    // Four-Stage Pulses.
    while (!active_nodes.empty()) {
        /*
            empty stash vars
        */
        memset(stash_stash_excess, 0, N*num_threads*sizeof(int64_t));

        // Stage 1: push.

        get_divide(num_threads, active_nodes.size(), disp, cnt);

        args_pool_stage1.clear();
        for(int i=0;i<num_threads;i++){
            args_pool_stage1.emplace_back(Stage1Args{N, i, cap, flow, dist, excess, stash_excess, stash_stash_excess+N*i, &active_nodes, &disp, &cnt, &inverse_flow_pool[i]});
        }

        pthread_barrier_init(&flow_write_barrier, NULL, num_threads);
        Thread::run_stage_1_task(args_pool_stage1);
        pthread_barrier_destroy(&flow_write_barrier);

        // Stage 2: relabel (update dist to stash_dist).
        memcpy(stash_dist, dist, N * sizeof(int));
        active_nodes_with_excess.clear();
        for(auto u: active_nodes){
            if(excess[u] > 0){
                active_nodes_with_excess.emplace_back(u);
            }
        }
        get_divide(num_threads, active_nodes_with_excess.size(), disp, cnt);
        args_pool_stage2.clear();
        for(int i=0;i<num_threads;i++){
            args_pool_stage2.emplace_back(Stage2Args{N, i, cap, flow, dist, stash_dist, excess, &active_nodes_with_excess, &disp, &cnt});
        }
        Thread::run_stage_2_task(args_pool_stage2);
        // for (auto u : active_nodes) {
        //     if (excess[u] > 0) {
        //         int min_dist = INT32_MAX;
        //         for (auto v = 0; v < N; v++) {
        //             auto residual_cap = cap[utils::idx(u, v, N)] - flow[utils::idx(u, v, N)];
        //             if (residual_cap > 0) {
        //                 min_dist = min(min_dist, dist[v]);
        //                 stash_dist[u] = min_dist + 1;
        //             }
        //         }
        //     }
        // }

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
