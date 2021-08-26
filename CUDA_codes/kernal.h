#define Pmax 1.0f
#define Imax 3.0f
#define delta 0.2f
#define input_file "M2C.txt"
#define output_file "C2M.txt"
const int num_user = 30;
const int num_PU = 5;
const int num_sub = 48;
const int num_rb = num_user * num_sub;
const int num_promising = 2;
const int num_subproblem =4096;
const int num_sub_blk =2;
const int L = 12; // correlation interval
const int Kx = 128;
const int Kp = num_subproblem/Kx;

const int BLOCK_SIZE = 2*num_sub*num_sub_blk;
const int GRID_SIZE = num_subproblem/ num_sub_blk;

__global__ void find_promising(float * config, float * criterion, int * promising_set);
__global__ void g_optimization(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball);
__global__ void final_result(float * current_best);
__global__ void g_optimization_old(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball);
__global__ void g_optimization_UB(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball);
