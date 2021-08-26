#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernal.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono> 

__global__ void find_promising(float * config, float * criterion, int * promising_set)
{
	int thread_idx = threadIdx.x;
	int block_idx = blockIdx.x;
	int tb_idx = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ float local_criterion[num_user];
	__shared__ int local_rb_index[num_user];
	__shared__ int best;
	criterion[tb_idx + num_rb] = config[tb_idx] * config[tb_idx + num_rb];
	float cri_norm = criterion[tb_idx + num_rb] / config[tb_idx + (num_PU * (2 + L) + 2)*num_rb];
	local_criterion[thread_idx] = cri_norm;
	local_rb_index[thread_idx] = tb_idx;
	criterion[tb_idx] = cri_norm;
	__syncthreads();
	//first time reduction
	unsigned int	s = num_user / 2;
	unsigned int	h = num_user;
	while (s > 1)
	{
		if (thread_idx < (h - s))
		{
			if (local_criterion[thread_idx] < local_criterion[thread_idx + s])
			{
				local_criterion[thread_idx] = local_criterion[thread_idx + s];
				local_rb_index[thread_idx] = local_rb_index[thread_idx + s];
			}
		}
		h = s;
		s = ceilf(s / 2.0);
	}
	if (thread_idx == 0)
	{
		if (local_criterion[thread_idx] < local_criterion[thread_idx + 1])
		{
			best = local_rb_index[1];
		}
		else
		{
			best = local_rb_index[0];
		}
	}
	__syncthreads();

	// Second time reduction
	if (tb_idx == best)
	{
		local_criterion[thread_idx] = 0.0f;
		local_rb_index[thread_idx] = tb_idx;
	}
	else
	{
		local_criterion[thread_idx] = cri_norm;
		local_rb_index[thread_idx] = tb_idx;
	}
	s = num_user / 2;
	h = num_user;
	while (s > 1)
	{
		if (thread_idx < (h - s))
		{
			if (local_criterion[thread_idx] < local_criterion[thread_idx + s])
			{
				local_criterion[thread_idx] = local_criterion[thread_idx + s];
				local_rb_index[thread_idx] = local_rb_index[thread_idx + s];
			}
		}
		h = s;
		s = ceilf(s / 2.0);
	}
	if (thread_idx == 0) {
		if (local_criterion[0] < local_criterion[1])
		{
			promising_set[block_idx*num_promising + 1] = local_rb_index[1];
		}
		else
		{
			promising_set[block_idx*num_promising + 1] = local_rb_index[0];
		}
	}
	else if (thread_idx == 1)
	{
		promising_set[block_idx*num_promising] = best;
	}
}

//TODO solve the optimization problem, original solution + refinement
__global__ void g_optimization(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball)
{
	// indexes with threads
	int block_idx = blockIdx.x;
	int thread_idx = threadIdx.x;
	int tb_idx = blockIdx.x*num_sub*num_sub_blk + threadIdx.x;
	int sub_idx = thread_idx % num_sub;
	//Shared memory
	__shared__ int sch_idx[num_sub*num_sub_blk];
	__shared__ int user_idx[num_sub*num_sub_blk];

	__shared__ float power_threshold[num_sub * 2 * num_sub_blk]; // power threshold of each user, 3 times faster
	__shared__ float power_assign[num_sub*num_sub_blk]; // power threshold of each user

	__shared__ float local_wh_norm[num_sub*num_sub_blk]; // w*h/mean_mu
	__shared__ float local_wh_sum[num_sub*num_sub_blk]; // sum_w*h

	__shared__ float interf_var[num_sub*num_PU*num_sub_blk];
	__shared__ float interf_mean[num_sub*num_PU*num_sub_blk];

	__shared__ float interf_flag[num_sub*num_PU*num_sub_blk];

	__shared__ float bound_flag[num_sub*num_sub_blk];

	if (thread_idx < num_sub*num_sub_blk)
	{
		float sum_criterion = 0.0f;
		sum_criterion = sampling_result[tb_idx] * (g_wh_criterion[promising_set[sub_idx*num_promising]] + g_wh_criterion[promising_set[sub_idx*num_promising + 1]]);
		if (block_idx < Kp / num_sub_blk)
		{
			sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
		}
		else
		{
			if (sum_criterion < g_wh_criterion[promising_set[sub_idx*num_promising]])
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
			}
			else
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising + 1];
			}
		}
	}
	else
	{
		bound_flag[thread_idx - num_sub * num_sub_blk] = 0.0f;
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk)
	{
		local_wh_sum[thread_idx] = g_wh_criterion[num_rb + sch_idx[thread_idx]];
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk)
	{
		local_wh_norm[thread_idx - num_sub * num_sub_blk] = g_wh_criterion[sch_idx[thread_idx - num_sub * num_sub_blk]];
	}
	// Reduction for sum of w.*h	
	__syncthreads();
	short int s = num_sub / 2;
	short int h = num_sub;
	while (s > 1)
	{
		if (sub_idx < (h - s) && thread_idx < num_sub*num_sub_blk)
		{
			local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + s];
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}

	if (sub_idx == 0 && thread_idx < num_sub*num_sub_blk)
	{
		local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + 1];
	}
	else if (thread_idx >= num_sub*num_sub_blk) //< 2*num_sub*num_sub_blk
	{
		for (int i = 0; i < 2; i++)
		{
			power_threshold[thread_idx - num_sub * num_sub_blk + i * num_sub*num_sub_blk] = 0.0f; //initailize the power threshold
		}
		user_idx[thread_idx - num_sub * num_sub_blk] = sch_idx[thread_idx - num_sub * num_sub_blk] % num_user;
	}
	__syncthreads();

	// K_p starting points
	if (thread_idx < num_sub*num_sub_blk)
	{
		if ((block_idx % (Kp / num_sub_blk) == 0) && thread_idx < num_sub)
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx];
		}
		else
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx] + delta * g_rand_ball[tb_idx];
		}
	}
	__syncthreads();

	// for calculate the total number of each subcarrier
	if (thread_idx < num_sub*num_sub_blk) //first half
	{
		for (int i = 0; i < num_sub / 2; i++)
		{
			if (user_idx[thread_idx] == user_idx[thread_idx - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - sub_idx + i];
			}
		}
	}
	else if (thread_idx < num_sub*num_sub_blk * 2) //second half
	{
		for (int i = num_sub / 2; i < num_sub; i++)
		{
			if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - num_sub * num_sub_blk - sub_idx + i];
			}
		}
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk) //obtain total threshold
	{
		power_threshold[thread_idx] = (power_threshold[thread_idx] + power_threshold[thread_idx + num_sub * num_sub_blk]);
		if (power_threshold[thread_idx] > Pmax)
		{
			power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
			bound_flag[thread_idx] = 1.0f;
			power_threshold[thread_idx] = Pmax;
		}
	}
	__syncthreads();
	// calculate interference

	if (thread_idx < num_sub*num_sub_blk) // calculate interference
	{
		for (int PU = 0; PU < num_PU / 2; PU++)
		{
			interf_var[thread_idx + PU * num_sub*num_sub_blk] = 0.0f;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr) //Same user has correlation
				{
					if (user_idx[thread_idx] == user_idx[thread_idx + corr])
					{
						interf_var[thread_idx + PU * num_sub*num_sub_blk] += power_assign[thread_idx] * power_assign[thread_idx + corr] * config[num_rb * ((L + 2)*PU + 2 + corr) + sch_idx[thread_idx]];
					}
				}
			}
			//interf_var[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 2) + sch_idx[thread_idx]];
			interf_mean[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx]];

			interf_flag[thread_idx + PU * num_sub*num_sub_blk] = interf_mean[thread_idx + PU * num_sub*num_sub_blk] * bound_flag[thread_idx];
		}
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk) //calculate interference_flag
	{
		for (int PU = num_PU / 2; PU < num_PU; PU++)
		{
			interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = 0;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr)//Same user has correlation
				{
					if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk + corr])
					{
						interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] += power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk + corr] * config[num_rb * (PU *(L + 2) + 2 + corr) + sch_idx[thread_idx - num_sub * num_sub_blk]];
					}
				}
			}
			//interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * (PU *2 + 2) + sch_idx[thread_idx - num_sub * num_sub_blk]];
			interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx - num_sub * num_sub_blk]];

			interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk] = interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] * bound_flag[thread_idx - num_sub * num_sub_blk];
		}
	}
	__syncthreads();
	// calculating sum of interference
	// need to calculate interf_var, interf_mean, interf_flag, total PU*2*3 parallel reduction

	//Loop 1, cut half, 8 batch per line
	int para_reduce_idx = thread_idx % (num_sub / 2);
	int para_reduce_tile = int(thread_idx / (num_sub / 2));
	interf_var[para_reduce_tile*num_sub + para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	interf_mean[para_reduce_tile*num_sub + para_reduce_idx] += interf_mean[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	interf_flag[para_reduce_tile*num_sub + para_reduce_idx] += interf_flag[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	if (para_reduce_tile<2)
	{
		interf_var[(8 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_var[(8 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	else if (para_reduce_tile<4)
	{
		interf_mean[(6 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_mean[(6 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	else if (para_reduce_tile<6)
	{
		interf_flag[(4 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_flag[(4 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	__syncthreads();
	//Loop 2: cut half, 16 batch per line
	para_reduce_idx = thread_idx % (num_sub / 4);
	para_reduce_tile = int(thread_idx / (num_sub / 4));
	if (para_reduce_tile<10)
	{
		interf_var[para_reduce_tile*num_sub + para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 4];
		interf_mean[para_reduce_tile*num_sub + para_reduce_idx] += interf_mean[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 4];
	}
	else if (para_reduce_tile<15)
	{
		interf_flag[(para_reduce_tile - 10)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 10)*num_sub + para_reduce_idx + num_sub / 4];
		interf_flag[(para_reduce_tile - 5)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 5)*num_sub + para_reduce_idx + num_sub / 4];
	}
	__syncthreads();
	//Loop 3: parallel reduction, 32 batch, only use 30
	
	para_reduce_idx = thread_idx % (num_sub / 8);
	para_reduce_tile = int(thread_idx / (num_sub / 8));
	s = num_sub / 8;
	h = num_sub / 4;
	while (s > 1)
	{
		if (para_reduce_idx < (h - s))
		{
			if (para_reduce_tile < 10)
			{
				interf_var[para_reduce_tile*num_sub+ para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + s];
			}
			else if (para_reduce_tile < 20)
			{
				interf_mean[(para_reduce_tile - 10)*num_sub + para_reduce_idx] += interf_mean[(para_reduce_tile - 10)*num_sub + para_reduce_idx + s];
			}
			else if (para_reduce_tile < 30)
			{
				interf_flag[(para_reduce_tile - 20)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 20)*num_sub + para_reduce_idx + s];
			}
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}
	if (thread_idx < 10)
	{
		interf_var[thread_idx*num_sub] += interf_var[thread_idx*num_sub + 1];
	}
	else if (thread_idx < 20)
	{
		interf_mean[(thread_idx - 10)*num_sub] += interf_mean[(thread_idx - 10)*num_sub + 1];
	}
	else if (thread_idx < 30)
	{
		interf_flag[(thread_idx - 20)*num_sub] += interf_flag[(thread_idx - 20)*num_sub + 1];
	}
	
	__syncthreads();

	// find maximum interference
	if (thread_idx < num_PU * 2)
	{
		interf_var[num_sub*thread_idx] = sqrtf(interf_var[num_sub*thread_idx]) + interf_mean[num_sub*thread_idx];
		if (interf_var[num_sub*thread_idx] > Imax)
		{
			interf_var[num_sub*thread_idx] = Imax / interf_var[num_sub*thread_idx];
		}
		else //need to scale up
		{
			interf_var[num_sub*thread_idx] = (Imax - interf_flag[num_sub*thread_idx]) / (interf_var[num_sub*thread_idx] - interf_flag[num_sub*thread_idx]);
		}
	}
	__syncthreads();
	if (thread_idx<4) //compare 0~3 with 6~9
	{
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 6)])
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 6)];
		}
	}
	if (thread_idx<2)
	{
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 4)]) //comare 0~1 with 4~5
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 4)];
		}
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 2)]) //comare 0~1 with 2~3
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 2)];
		}
	}
	__syncthreads();


	if (thread_idx < num_sub*num_sub_blk) //first half
	{

		if (interf_var[thread_idx - sub_idx] > 1.0f) //scaling up
		{
			if (bound_flag[thread_idx] == 0.0f)
			{
				power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx];
				power_threshold[thread_idx] = power_threshold[thread_idx] * interf_var[thread_idx - sub_idx];
				if (power_threshold[thread_idx] > Pmax)
				{
					power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
				}
			}
		}
		else
		{
			power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx]; //scaling down
		}

		interf_var[thread_idx] = config[num_rb + sch_idx[thread_idx]] * log2f(1.0f + config[sch_idx[thread_idx]] * power_assign[thread_idx]);// interference for objective
	}
	__syncthreads();
	s = num_sub / 2;
	h = num_sub;
	while (s > 1)
	{
		if (sub_idx < (h - s) && thread_idx < num_sub * num_sub_blk)
		{
			interf_var[thread_idx] += interf_var[thread_idx + s];
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}
	if (sub_idx == 0 && thread_idx < num_sub * num_sub_blk)
	{
		interf_var[thread_idx] = interf_var[thread_idx] + interf_var[thread_idx + 1]; //objective for each sub-problem
	}
	__syncthreads();
	__shared__ int best_index_blk; //highest in current thredblock
	if (thread_idx == 0)
	{
		if (interf_var[0] < interf_var[num_sub]) //second subproblem higher
		{
			best_index_blk = 1;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[num_sub];
		}
		else //first subproblem higher
		{
			best_index_blk = 0;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[0];
		}
	}
	__syncthreads();
	//copy the solution to global memory
	if (thread_idx < num_sub)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = sch_idx[best_index_blk * num_sub + thread_idx]; //power assignment
	}
	else if (thread_idx < num_sub * 2)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = power_assign[best_index_blk * num_sub - num_sub + thread_idx]; //scheduling decision
	}
}


//TODO compare for the best result
__global__ void final_result(float * current_best)
{
	int thread_idx = threadIdx.x;
	//int block_idx = blockIdx.x;
	__shared__ short int global_index[GRID_SIZE/2];
	__shared__ float local_best[GRID_SIZE/2];
	// * reduction to find global results
	if (current_best[thread_idx*(2 * num_sub + 1)] < current_best[thread_idx*(2 * num_sub + 1) + GRID_SIZE/2 * (2 * num_sub + 1)])
	{
		local_best[thread_idx] = current_best[thread_idx*(2 * num_sub + 1) + GRID_SIZE/2 * (2 * num_sub + 1)];
		global_index[thread_idx] = thread_idx + GRID_SIZE/2;
	}
	else
	{
		local_best[thread_idx] = current_best[thread_idx*(2 * num_sub + 1)];
		global_index[thread_idx] = thread_idx;
	}
	short int	s = GRID_SIZE/4;
	short int	h = GRID_SIZE/2;
	__syncthreads();
	while (s > 1)
	{
		if (thread_idx < (h - s))
		{
			if (local_best[thread_idx] < local_best[thread_idx + s])
			{
				local_best[thread_idx] = local_best[thread_idx + s];
				global_index[thread_idx] = global_index[thread_idx + s];
			}
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}
	if (thread_idx == 0)
	{
		if (local_best[thread_idx] < local_best[thread_idx + 1])
		{
			current_best[0] = local_best[thread_idx + 1];
			global_index[thread_idx] = global_index[thread_idx + 1];
		}
		else
		{
			current_best[0] = local_best[thread_idx];
			global_index[thread_idx + 1] = global_index[thread_idx];
		}
	}
	__syncthreads();
	// trans the best solution or index
	if (thread_idx < num_sub)
	{
		current_best[thread_idx + 1] = current_best[global_index[0] * (2 * num_sub + +1) + thread_idx + 1]; //final scheduling decision
		current_best[thread_idx + num_sub + 1] = current_best[global_index[1] * (2 * num_sub + 1) + num_sub + thread_idx + 1];  //final power allocation
	}
}


//TODO solve the optimization problem, original solution + refinement
__global__ void g_optimization_old(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball)
{
	// indexes with threads
	int block_idx = blockIdx.x;
	int thread_idx = threadIdx.x;
	int tb_idx = blockIdx.x*num_sub*num_sub_blk + threadIdx.x;
	int sub_idx = thread_idx % num_sub;
	//Shared memory
	__shared__ int sch_idx[num_sub*num_sub_blk];
	__shared__ int user_idx[num_sub*num_sub_blk];

	__shared__ float power_threshold[num_sub * 2 * num_sub_blk]; // power threshold of each user, 3 times faster
	__shared__ float power_assign[num_sub*num_sub_blk]; // power threshold of each user

	__shared__ float local_wh_norm[num_sub*num_sub_blk]; // w*h/mean_mu
	__shared__ float local_wh_sum[num_sub*num_sub_blk]; // sum_w*h

	__shared__ float interf_var[num_sub*num_PU*num_sub_blk];
	__shared__ float interf_mean[num_sub*num_PU*num_sub_blk];

	__shared__ float interf_flag[num_sub*num_PU*num_sub_blk];

	__shared__ float bound_flag[num_sub*num_sub_blk];

	if (thread_idx < num_sub*num_sub_blk)
	{
		float sum_criterion = 0.0f;
		sum_criterion = sampling_result[tb_idx] * (g_wh_criterion[promising_set[sub_idx*num_promising]] + g_wh_criterion[promising_set[sub_idx*num_promising + 1]]);
		if (block_idx < Kp / num_sub_blk)
		{
			sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
		}
		else
		{
			if (sum_criterion < g_wh_criterion[promising_set[sub_idx*num_promising]])
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
			}
			else
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising + 1];
			}
		}
	}
	else
	{
		bound_flag[thread_idx - num_sub * num_sub_blk] = 0.0f;
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk)
	{
		local_wh_sum[thread_idx] = g_wh_criterion[num_rb + sch_idx[thread_idx]];
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk)
	{
		local_wh_norm[thread_idx - num_sub * num_sub_blk] = g_wh_criterion[sch_idx[thread_idx - num_sub * num_sub_blk]];
	}
	// Reduction for sum of w.*h	
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk) // 0~num_sub-1 for reduction sum
	{
		short int s = num_sub / 2;
		short int h = num_sub;
		while (s > 1)
		{
			if (sub_idx < (h - s))
			{
				local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + s];
			}
			h = s;
			s = ceilf(s / 2.0);
		}
		if (sub_idx == 0)
		{
			local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + 1];
		}
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk)
	{
		for (int i = 0; i < 2; i++)
		{
			power_threshold[thread_idx - num_sub * num_sub_blk + i * num_sub*num_sub_blk] = 0.0f; //initailize the power threshold
		}
		user_idx[thread_idx - num_sub * num_sub_blk] = sch_idx[thread_idx - num_sub * num_sub_blk] % num_user;
	}
	__syncthreads();

	// K_p starting points
	//	for (int p_index = 0; p_index < K_p; p_index++)
	//{
	if (thread_idx < num_sub*num_sub_blk)
	{
		if ((block_idx % (Kp / num_sub_blk) == 0) && thread_idx < num_sub)
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx];
		}
		else
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx] + delta * g_rand_ball[tb_idx];
		}
	}
	__syncthreads();

	// for calculate the total number of each subcarrier
	if (thread_idx < num_sub*num_sub_blk) //first half
	{
		for (int i = 0; i < num_sub / 2; i++)
		{
			if (user_idx[thread_idx] == user_idx[thread_idx - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - sub_idx + i];
			}
		}
	}
	else if (thread_idx < num_sub*num_sub_blk * 2) //second half
	{
		for (int i = num_sub / 2; i < num_sub; i++)
		{
			if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - num_sub * num_sub_blk - sub_idx + i];
			}
		}
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk) //obtain total threshold
	{
		power_threshold[thread_idx] = (power_threshold[thread_idx] + power_threshold[thread_idx + num_sub * num_sub_blk]);
		if (power_threshold[thread_idx] > Pmax)
		{
			power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
			bound_flag[thread_idx] = 1.0f;
			power_threshold[thread_idx] = Pmax;
		}
	}
	__syncthreads();
	// calculate interference

	if (thread_idx < num_sub*num_sub_blk) // calculate interference
	{
		for (int PU = 0; PU < num_PU / 2; PU++)
		{
			interf_var[thread_idx + PU * num_sub*num_sub_blk] = 0.0f;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr) //Same user has correlation
				{
					if (user_idx[thread_idx] == user_idx[thread_idx + corr])
					{
						interf_var[thread_idx + PU * num_sub*num_sub_blk] += power_assign[thread_idx] * power_assign[thread_idx + corr] * config[num_rb * ((L + 2)*PU + 2 + corr) + sch_idx[thread_idx]];
					}
				}
			}
			//interf_var[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 2) + sch_idx[thread_idx]];
			interf_mean[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx]];

			interf_flag[thread_idx + PU * num_sub*num_sub_blk] = interf_mean[thread_idx + PU * num_sub*num_sub_blk] * bound_flag[thread_idx];
		}
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk) //calculate interference_flag
	{
		for (int PU = num_PU / 2; PU < num_PU; PU++)
		{
			interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = 0;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr)//Same user has correlation
				{
					if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk + corr])
					{
						interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] += power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk + corr] * config[num_rb * (PU *(L + 2) + 2 + corr) + sch_idx[thread_idx - num_sub * num_sub_blk]];
					}
				}
			}
			//interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * (PU *2 + 2) + sch_idx[thread_idx - num_sub * num_sub_blk]];
			interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx - num_sub * num_sub_blk]];

			interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk] = interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] * bound_flag[thread_idx - num_sub * num_sub_blk];
		}
	}
	__syncthreads();
	// calculating sum of interference
	if (thread_idx < num_sub*num_sub_blk) //first half for sum of total interference
	{
		if (sub_idx < num_sub / 2) {
			short int s = num_sub / 2;
			short int h = num_sub;
			while (s > 1)
			{
				if (sub_idx < (h - s))
				{
					for (int PU = 0; PU < num_PU - 1; PU++)
					{
						interf_var[thread_idx + PU * num_sub*num_sub_blk] += interf_var[thread_idx + PU * num_sub*num_sub_blk + s];
					}
				}
				h = s;
				s = ceilf(s / 2.0);
			}
			if (sub_idx == 0)
			{
				for (int PU = 0; PU < num_PU - 1; PU++)
				{
					interf_var[thread_idx + PU * num_sub*num_sub_blk] += interf_var[thread_idx + PU * num_sub*num_sub_blk + 1];
				}
			}
		}
		else if (sub_idx < num_sub)
		{
			short int s = num_sub / 2;
			short int h = num_sub;
			while (s > 1)
			{
				if (sub_idx - num_sub / 2 < (h - s))
				{
					for (int PU = 0; PU < num_PU - 1; PU++)
					{
						interf_mean[thread_idx - num_sub / 2 + PU * num_sub*num_sub_blk] += interf_mean[thread_idx - num_sub / 2 + PU * num_sub*num_sub_blk + s];
					}
				}
				h = s;
				s = ceilf(s / 2.0);
			}
			if (sub_idx == num_sub / 2)
			{
				for (int PU = 0; PU < num_PU - 1; PU++)
				{
					interf_mean[thread_idx - num_sub / 2 + PU * num_sub*num_sub_blk] += interf_mean[thread_idx - num_sub / 2 + PU * num_sub*num_sub_blk + 1];
				}
			}
		}
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk) //first half for sum of flagged partial total interference
	{
		if (sub_idx < num_sub / 2) {
			short int s = num_sub / 2;
			short int h = num_sub;
			while (s > 1)
			{
				if (sub_idx < (h - s))
				{
					for (int PU = 0; PU < num_PU - 1; PU++)
					{
						interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk] += interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk + s];
					}
				}
				h = s;
				s = ceilf(s / 2.0);
			}
			if (sub_idx == 0)
			{
				for (int PU = 0; PU < num_PU - 1; PU++)
				{
					interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk] += interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk + 1];
				}
			}
		}
		else if (sub_idx < num_sub) //The last PU
		{
			short int s = num_sub / 2;
			short int h = num_sub;
			while (s > 1)
			{
				if (sub_idx - num_sub / 2 < (h - s))
				{
					interf_var[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_var[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + s];
					interf_mean[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_mean[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + s];
					interf_flag[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_flag[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + s];
				}
				h = s;
				s = ceilf(s / 2.0);
			}
			if (sub_idx == num_sub / 2)
			{
				interf_var[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_var[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + 1];
				interf_mean[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_mean[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + 1];
				interf_flag[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk] += interf_flag[thread_idx - num_sub / 2 + (num_PU - 2) * num_sub*num_sub_blk + 1];
			}
		}
	}
	__syncthreads();

	// find maximum interference
	if (thread_idx < num_sub*num_sub_blk)
	{
		if (sub_idx < num_PU)
		{
			// Calculate modified interference
			interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] = sqrtf(interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)]) + interf_mean[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)];
			if (interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] > Imax) // need to scale down
			{
				interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] = Imax / interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)];
			}
			else //need to scale up
			{
				interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] = (Imax - interf_flag[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)]) / (interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] - interf_flag[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)]);
			}
		}
		short int s = ceilf(num_PU / 2.0);
		short int h = num_PU;
		while (s > 1)
		{
			if (sub_idx < (h - s))
			{
				if (interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] > interf_var[num_sub*((sub_idx + s)*num_sub_blk + thread_idx / num_sub)])
				{
					interf_var[num_sub*(sub_idx*num_sub_blk + thread_idx / num_sub)] = interf_var[num_sub*((sub_idx + s)*num_sub_blk + thread_idx / num_sub)];
				}
			}
			h = s;
			s = ceilf(s / 2.0);
		}
		if (sub_idx == 0)
		{
			if (interf_var[thread_idx] > interf_var[thread_idx + num_sub * num_sub_blk])
			{
				interf_var[thread_idx] = interf_var[thread_idx + num_sub * num_sub_blk];
			}
		}
	}
	__syncthreads();

	if (thread_idx < num_sub*num_sub_blk) //first half
	{

		if (interf_var[thread_idx - sub_idx] > 1.0f) //scaling up
		{
			if (bound_flag[thread_idx] == 0.0f)
			{
				power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx];
				power_threshold[thread_idx] = power_threshold[thread_idx] * interf_var[thread_idx - sub_idx];
				if (power_threshold[thread_idx] > Pmax)
				{
					power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
				}
			}
		}
		else
		{
			power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx]; //scaling down
		}

		interf_var[thread_idx] = config[num_rb + sch_idx[thread_idx]] * log2f(1.0f + config[sch_idx[thread_idx]] * power_assign[thread_idx]);// interference for objective
	}
	__syncthreads();
	if (thread_idx < num_sub * num_sub_blk)
	{
		short int s = num_sub / 2;
		short int h = num_sub;
		while (s > 1)
		{
			if (sub_idx < (h - s))
			{
				interf_var[thread_idx] += interf_var[thread_idx + s];
			}
			h = s;
			s = ceilf(s / 2.0);
		}
		if (sub_idx == 0)
		{
			interf_var[thread_idx] = interf_var[thread_idx] + interf_var[thread_idx + 1]; //objective for each sub-problem
		}
	}
	__syncthreads();
	__shared__ int best_index_blk; //highest in current thredblock
	if (thread_idx == 0)
	{
		if (interf_var[0] < interf_var[num_sub]) //second subproblem higher
		{
			best_index_blk = 1;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[num_sub];
		}
		else //first subproblem higher
		{
			best_index_blk = 0;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[0];
		}
	}
	__syncthreads();
	//copy the solution to global memory
	if (thread_idx < num_sub)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = sch_idx[best_index_blk * num_sub + thread_idx]; //power assignment
	}
	else if (thread_idx < num_sub * 2)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = power_assign[best_index_blk * num_sub - num_sub + thread_idx]; //scheduling decision
	}
}

//TODO solve the optimization problem, original solution + refinement
__global__ void g_optimization_UB(float * config, float * current_best, int * promising_set, float* sampling_result, float * g_wh_criterion, float * g_rand_ball)
{
	// indexes with threads
	int block_idx = blockIdx.x;
	int thread_idx = threadIdx.x;
	int tb_idx = blockIdx.x*num_sub*num_sub_blk + threadIdx.x;
	int sub_idx = thread_idx % num_sub;
	//Shared memory
	__shared__ int sch_idx[num_sub*num_sub_blk];
	__shared__ int user_idx[num_sub*num_sub_blk];

	__shared__ float power_threshold[num_sub * 2 * num_sub_blk]; // power threshold of each user, 3 times faster
	__shared__ float power_assign[num_sub*num_sub_blk]; // power threshold of each user

	__shared__ float local_wh_norm[num_sub*num_sub_blk]; // w*h/mean_mu
	__shared__ float local_wh_sum[num_sub*num_sub_blk]; // sum_w*h

	__shared__ float interf_var[num_sub*num_PU*num_sub_blk];
	__shared__ float interf_mean[num_sub*num_PU*num_sub_blk];

	__shared__ float interf_flag[num_sub*num_PU*num_sub_blk];

	__shared__ float bound_flag[num_sub*num_sub_blk];

	if (thread_idx < num_sub*num_sub_blk)
	{
		float sum_criterion = 0.0f;
		sum_criterion = sampling_result[tb_idx] * (g_wh_criterion[promising_set[sub_idx*num_promising]] + g_wh_criterion[promising_set[sub_idx*num_promising + 1]]);
		if (block_idx < Kp / num_sub_blk)
		{
			sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
		}
		else
		{
			if (sum_criterion < g_wh_criterion[promising_set[sub_idx*num_promising]])
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising];
			}
			else
			{
				sch_idx[thread_idx] = promising_set[sub_idx*num_promising + 1];
			}
		}
	}
	else
	{
		bound_flag[thread_idx - num_sub * num_sub_blk] = 0.0f;
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk)
	{
		local_wh_sum[thread_idx] = g_wh_criterion[num_rb + sch_idx[thread_idx]];
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk)
	{
		local_wh_norm[thread_idx - num_sub * num_sub_blk] = g_wh_criterion[sch_idx[thread_idx - num_sub * num_sub_blk]];
	}
	// Reduction for sum of w.*h	
	__syncthreads();
	short int s = num_sub / 2;
	short int h = num_sub;
	while (s > 1)
	{
		if (sub_idx < (h - s) && thread_idx < num_sub*num_sub_blk)
		{
			local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + s];
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}

	if (sub_idx == 0 && thread_idx < num_sub*num_sub_blk)
	{
		local_wh_sum[thread_idx] = local_wh_sum[thread_idx] + local_wh_sum[thread_idx + 1];
	}
	else if (thread_idx >= num_sub*num_sub_blk) //< 2*num_sub*num_sub_blk
	{
		for (int i = 0; i < 2; i++)
		{
			power_threshold[thread_idx - num_sub * num_sub_blk + i * num_sub*num_sub_blk] = 0.0f; //initailize the power threshold
		}
		user_idx[thread_idx - num_sub * num_sub_blk] = sch_idx[thread_idx - num_sub * num_sub_blk] % num_user;
	}
	__syncthreads();

	// K_p starting points
	if (thread_idx < num_sub*num_sub_blk)
	{
		if ((block_idx % (Kp / num_sub_blk) == 0) && thread_idx < num_sub)
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx];
		}
		else
		{
			power_assign[thread_idx] = Imax * local_wh_norm[thread_idx] / local_wh_sum[thread_idx - sub_idx] + delta * g_rand_ball[tb_idx];
		}
	}
	__syncthreads();

	// for calculate the total number of each subcarrier
	if (thread_idx < num_sub*num_sub_blk) //first half
	{
		for (int i = 0; i < num_sub / 2; i++)
		{
			if (user_idx[thread_idx] == user_idx[thread_idx - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - sub_idx + i];
			}
		}
	}
	else if (thread_idx < num_sub*num_sub_blk * 2) //second half
	{
		for (int i = num_sub / 2; i < num_sub; i++)
		{
			if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk - sub_idx + i])
			{
				power_threshold[thread_idx] = power_threshold[thread_idx] + power_assign[thread_idx - num_sub * num_sub_blk - sub_idx + i];
			}
		}
	}
	__syncthreads();
	if (thread_idx < num_sub*num_sub_blk) //obtain total threshold
	{
		power_threshold[thread_idx] = (power_threshold[thread_idx] + power_threshold[thread_idx + num_sub * num_sub_blk]);
		if (power_threshold[thread_idx] > Pmax)
		{
			power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
			bound_flag[thread_idx] = 1.0f;
			power_threshold[thread_idx] = Pmax;
		}
	}
	__syncthreads();
	// calculate interference

	if (thread_idx < num_sub*num_sub_blk) // calculate interference
	{
		for (int PU = 0; PU < num_PU / 2; PU++)
		{
			interf_var[thread_idx + PU * num_sub*num_sub_blk] = 0.0f;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr) //Same user has correlation
				{
					//if (user_idx[thread_idx] == user_idx[thread_idx + corr])
					//{
						interf_var[thread_idx + PU * num_sub*num_sub_blk] += power_assign[thread_idx] * power_assign[thread_idx + corr] * config[num_rb * ((L + 2)*PU + 2 + corr) + sch_idx[thread_idx]];
					//}
				}
			}
			//interf_var[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 2) + sch_idx[thread_idx]];
			interf_mean[thread_idx + PU * num_sub*num_sub_blk] = power_assign[thread_idx] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx]];

			interf_flag[thread_idx + PU * num_sub*num_sub_blk] = interf_mean[thread_idx + PU * num_sub*num_sub_blk] * bound_flag[thread_idx];
		}
	}
	else if (thread_idx < 2 * num_sub*num_sub_blk) //calculate interference_flag
	{
		for (int PU = num_PU / 2; PU < num_PU; PU++)
		{
			interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = 0;
			for (int corr = 0; corr < L + 1; corr++) //Calculate p^TRp
			{
				if (sub_idx < num_sub - corr)//Same user has correlation
				{
					//if (user_idx[thread_idx - num_sub * num_sub_blk] == user_idx[thread_idx - num_sub * num_sub_blk + corr])
					//{
						interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] += power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk + corr] * config[num_rb * (PU *(L + 2) + 2 + corr) + sch_idx[thread_idx - num_sub * num_sub_blk]];
					//}
				}
			}
			//interf_var[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * (PU *2 + 2) + sch_idx[thread_idx - num_sub * num_sub_blk]];
			interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] = power_assign[thread_idx - num_sub * num_sub_blk] * config[num_rb * ((L + 2)*PU + 3 + L) + sch_idx[thread_idx - num_sub * num_sub_blk]];

			interf_flag[thread_idx + (PU - 1) * num_sub*num_sub_blk] = interf_mean[thread_idx + (PU - 1) * num_sub*num_sub_blk] * bound_flag[thread_idx - num_sub * num_sub_blk];
		}
	}
	__syncthreads();
	// calculating sum of interference
	// need to calculate interf_var, interf_mean, interf_flag, total PU*2*3 parallel reduction

	//Loop 1, cut half, 8 batch per line
	int para_reduce_idx = thread_idx % (num_sub / 2);
	int para_reduce_tile = int(thread_idx / (num_sub / 2));
	interf_var[para_reduce_tile*num_sub + para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	interf_mean[para_reduce_tile*num_sub + para_reduce_idx] += interf_mean[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	interf_flag[para_reduce_tile*num_sub + para_reduce_idx] += interf_flag[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 2];
	if (para_reduce_tile<2)
	{
		interf_var[(8 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_var[(8 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	else if (para_reduce_tile<4)
	{
		interf_mean[(6 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_mean[(6 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	else if (para_reduce_tile<6)
	{
		interf_flag[(4 + para_reduce_tile)*num_sub + para_reduce_idx] += interf_flag[(4 + para_reduce_tile)*num_sub + para_reduce_idx + num_sub / 2];
	}
	__syncthreads();
	//Loop 2: cut half, 16 batch per line
	para_reduce_idx = thread_idx % (num_sub / 4);
	para_reduce_tile = int(thread_idx / (num_sub / 4));
	if (para_reduce_tile<10)
	{
		interf_var[para_reduce_tile*num_sub + para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 4];
		interf_mean[para_reduce_tile*num_sub + para_reduce_idx] += interf_mean[para_reduce_tile*num_sub + para_reduce_idx + num_sub / 4];
	}
	else if (para_reduce_tile<15)
	{
		interf_flag[(para_reduce_tile - 10)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 10)*num_sub + para_reduce_idx + num_sub / 4];
		interf_flag[(para_reduce_tile - 5)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 5)*num_sub + para_reduce_idx + num_sub / 4];
	}
	__syncthreads();
	//Loop 3: parallel reduction, 32 batch, only use 30
	
	para_reduce_idx = thread_idx % (num_sub / 8);
	para_reduce_tile = int(thread_idx / (num_sub / 8));
	s = num_sub / 8;
	h = num_sub / 4;
	while (s > 1)
	{
		if (para_reduce_idx < (h - s))
		{
			if (para_reduce_tile < 10)
			{
				interf_var[para_reduce_tile*num_sub+ para_reduce_idx] += interf_var[para_reduce_tile*num_sub + para_reduce_idx + s];
			}
			else if (para_reduce_tile < 20)
			{
				interf_mean[(para_reduce_tile - 10)*num_sub + para_reduce_idx] += interf_mean[(para_reduce_tile - 10)*num_sub + para_reduce_idx + s];
			}
			else if (para_reduce_tile < 30)
			{
				interf_flag[(para_reduce_tile - 20)*num_sub + para_reduce_idx] += interf_flag[(para_reduce_tile - 20)*num_sub + para_reduce_idx + s];
			}
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}
	if (thread_idx < 10)
	{
		interf_var[thread_idx*num_sub] += interf_var[thread_idx*num_sub + 1];
	}
	else if (thread_idx < 20)
	{
		interf_mean[(thread_idx - 10)*num_sub] += interf_mean[(thread_idx - 10)*num_sub + 1];
	}
	else if (thread_idx < 30)
	{
		interf_flag[(thread_idx - 20)*num_sub] += interf_flag[(thread_idx - 20)*num_sub + 1];
	}
	
	__syncthreads();

	// find maximum interference
	if (thread_idx < num_PU * 2)
	{
		interf_var[num_sub*thread_idx] = sqrtf(interf_var[num_sub*thread_idx]) + interf_mean[num_sub*thread_idx];
		if (interf_var[num_sub*thread_idx] > Imax)
		{
			interf_var[num_sub*thread_idx] = Imax / interf_var[num_sub*thread_idx];
		}
		else //need to scale up
		{
			interf_var[num_sub*thread_idx] = (Imax - interf_flag[num_sub*thread_idx]) / (interf_var[num_sub*thread_idx] - interf_flag[num_sub*thread_idx]);
		}
	}
	__syncthreads();
	if (thread_idx<4) //compare 0~3 with 6~9
	{
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 6)])
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 6)];
		}
	}
	if (thread_idx<2)
	{
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 4)]) //comare 0~1 with 4~5
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 4)];
		}
		if (interf_var[num_sub*thread_idx] > interf_var[num_sub*(thread_idx + 2)]) //comare 0~1 with 2~3
		{
			interf_var[num_sub*thread_idx] = interf_var[num_sub*(thread_idx + 2)];
		}
	}
	__syncthreads();


	if (thread_idx < num_sub*num_sub_blk) //first half
	{

		if (interf_var[thread_idx - sub_idx] > 1.0f) //scaling up
		{
			if (bound_flag[thread_idx] == 0.0f)
			{
				power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx];
				power_threshold[thread_idx] = power_threshold[thread_idx] * interf_var[thread_idx - sub_idx];
				if (power_threshold[thread_idx] > Pmax)
				{
					power_assign[thread_idx] = power_assign[thread_idx] * Pmax / power_threshold[thread_idx];
				}
			}
		}
		else
		{
			power_assign[thread_idx] = power_assign[thread_idx] * interf_var[thread_idx - sub_idx]; //scaling down
		}

		interf_var[thread_idx] = config[num_rb + sch_idx[thread_idx]] * log2f(1.0f + config[sch_idx[thread_idx]] * power_assign[thread_idx]);// interference for objective
	}
	__syncthreads();
	s = num_sub / 2;
	h = num_sub;
	while (s > 1)
	{
		if (sub_idx < (h - s) && thread_idx < num_sub * num_sub_blk)
		{
			interf_var[thread_idx] += interf_var[thread_idx + s];
		}
		h = s;
		s = ceilf(s / 2.0);
		__syncthreads();
	}
	if (sub_idx == 0 && thread_idx < num_sub * num_sub_blk)
	{
		interf_var[thread_idx] = interf_var[thread_idx] + interf_var[thread_idx + 1]; //objective for each sub-problem
	}
	__syncthreads();
	__shared__ int best_index_blk; //highest in current thredblock
	if (thread_idx == 0)
	{
		if (interf_var[0] < interf_var[num_sub]) //second subproblem higher
		{
			best_index_blk = 1;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[num_sub];
		}
		else //first subproblem higher
		{
			best_index_blk = 0;
			current_best[block_idx*(2 * num_sub + 1)] = interf_var[0];
		}
	}
	__syncthreads();
	//copy the solution to global memory
	if (thread_idx < num_sub)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = sch_idx[best_index_blk * num_sub + thread_idx]; //power assignment
	}
	else if (thread_idx < num_sub * 2)
	{
		current_best[block_idx*(2 * num_sub + 1) + thread_idx + 1] = power_assign[best_index_blk * num_sub - num_sub + thread_idx]; //scheduling decision
	}
}
