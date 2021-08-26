#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernal.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <string>
#include <algorithm>
#include <functional>
#include <array>
#include <vector>
#include <ctime>
using namespace std;

int main()
{
	cudaDeviceReset();
	
	cudaError_t error1 = cudaGetLastError();
	error1 = cudaSetDevice(0);

	//CPU variables
	float config[(num_PU*(L+2) + 3)*num_rb];
	float final_sch_power[1 + 2 * num_sub];
 	float *sampling_result= new float[num_subproblem*num_sub];
	float *rand_ball= new float[num_subproblem*num_sub];
	
	//GPU variables
	float *g_config, *g_current_best, *g_rand_ball;
	float * g_sampling_result, *g_wh_criterion;
	int  *g_promising_set;

	//Pre-process, not changed in each TTI
	//Read data from M2C.txt, including {h,w,mu,mean_mu}
	std::ifstream M2C(input_file);
	for (int i = 0; i < (num_PU*(L+2) + 3)*num_rb; i++)
	{
		M2C >> config[i];
		//config[i]+=0.0001f;
		//std::cout<<config[i]<<std::endl;
	}
	M2C.close();

	//srand(time(0));
	// Force the first subproblem to have the highest index
	for (int i = 0; i < num_subproblem*num_sub; i++)
	{
		if (i % (num_sub*Kp) < num_sub)
		{
			rand_ball[i] = 0.0f;
		}
		else
		{
			rand_ball[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}

		if (i % (num_sub*Kp) < num_sub)
		{
			if (i < num_sub)
			{
				sampling_result[i] = 0.0f;
			}
			else
			{
				sampling_result[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			}
		}
		else
		{
			sampling_result[i] = sampling_result[i - num_sub];
		}
	}

	// All subproblems are generated with random samplings
	// for (int i = 0; i < num_subproblem*num_sub; i++)
	// {
	// 	if (i % (num_sub*Kp) < num_sub)
	// 	{
	// 		rand_ball[i] = 0.0f;
	// 	}
	// 	else
	// 	{
	// 		rand_ball[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	// 	}

	// 	if (i % (num_sub*Kp) < num_sub)
	// 	{
	// 		if (i < num_sub)
	// 		{
	// 			sampling_result[i] = 0.0f;
	// 		}
	// 		else
	// 		{
	// 			sampling_result[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	// 		}
	// 	}
	// 	else
	// 	{
	// 		sampling_result[i] = sampling_result[i - num_sub];
	// 	}
	// }

	const int instance = 1;
	float elapsed_time[instance];
	for (int sta = 0; sta < instance; sta++)
	{
		// Allocate GPU buffers for vectors (two input, one output)    .
		cudaMalloc(&g_promising_set, num_promising*num_sub * sizeof(int));
		cudaMalloc(&g_config, (num_PU*(L+2) + 3)*num_rb * sizeof(float));
		cudaMalloc(&g_current_best, (2 * num_sub + 1)*num_subproblem/num_sub_blk * sizeof(float));
		cudaMalloc(&g_sampling_result, num_sub*num_subproblem * sizeof(int));
		cudaMalloc(&g_wh_criterion, 2 * num_rb * sizeof(float));
		cudaMalloc(&g_rand_ball, num_sub * num_subproblem * sizeof(float));

		cudaMemcpy(g_sampling_result, sampling_result, num_sub*num_subproblem * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(g_config + 2 * num_rb, config + 2 * num_rb, (num_PU*(L+2) + 1)*num_rb * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(g_rand_ball, rand_ball, num_sub * num_subproblem * sizeof(float), cudaMemcpyHostToDevice);

		float milliseconds = 0.0;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
		// Copy w and h to GPU
		cudaMemcpyAsync(g_config, config, 2 * num_rb * sizeof(float), cudaMemcpyHostToDevice);
		//TODO calculate criterion and find the promising
		find_promising << <num_sub, num_user >> > (g_config, g_wh_criterion, g_promising_set);
		//TODO solve the optimization problem, original solution + refinement
		g_optimization << <GRID_SIZE, BLOCK_SIZE >> > (g_config, g_current_best, g_promising_set, g_sampling_result, g_wh_criterion,g_rand_ball);

		//TODO compare for the best result
		final_result << <1, GRID_SIZE/2 >> > (g_current_best);
		// TODO print final solution
		cudaMemcpy(final_sch_power, g_current_best, (1 + 2 * num_sub) * sizeof(float), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);
		elapsed_time[sta] = milliseconds;

		cudaFree(g_config);
		cudaFree(g_current_best);
		cudaFree(g_sampling_result);
		cudaFree(g_wh_criterion);
		cudaFree(g_promising_set);
		cudaFree(g_rand_ball);
		cudaDeviceReset();
	}
	// Write data to C2M.txt, including p
	//cudaError_t errot = cudaGetLastError();
	//
	float obj = 0.0f;
//	for(int i=0;i<num_sub*2+1;i++)
//	{
//		std::cout<<final_sch_power[i]<<std::endl;
//	}
	for (int i = 0; i < num_sub; i++)
	{
		obj += config[num_rb + int(final_sch_power[i + 1])] * log2f(1.0f + config[int(final_sch_power[i + 1])] * final_sch_power[i + 1 + num_sub]);
	}
	std::cout<<obj<<std::endl;
	std::cout << final_sch_power[0] << std::endl;

	std::ofstream C2M(output_file);
	//float mean_time=0.0f;
	C2M << elapsed_time[0] << ' ';
	//std::cout<<mean_time/10000.0<<std::endl;
	for (int i = 0; i < 2*num_sub+1; i++)
	{
		C2M << final_sch_power[i] << ' ';
	}

	C2M.close();

	free(sampling_result);
	free(rand_ball);
	//system("pause");
	return 0;
}
