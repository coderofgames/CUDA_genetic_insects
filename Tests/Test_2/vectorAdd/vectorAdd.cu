/**
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
* Vector addition: C = A + B.
*
* This sample is a very basic sample that implements element by element
* vector addition. It is the same as the sample illustrating Chapter 2
* of the programming guide with some additions like error checking.
*/

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <cooperative_groups.h>
#include <helper_functions.h>


namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
	__device__ inline operator double *()
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}

	__device__ inline operator const double *() const
	{
		extern __shared__ double __smem_d[];
		return (double *)__smem_d;
	}
};


int chromo;
//int chromo;
//int chromo;
void SetBit(int &dna, int idx)
{
	dna |= (1 << idx);
}

bool CheckBit(int &dna, int idx)
{
	return dna & (1 << idx);
}

void ResetBit(int &dna, int idx)
{
	dna &= ~(1 << idx);
}


int ANTENNAE_MASK = 0;
int HEAD_MASK = 0;
int WINGS_MASK = 0;
int BODY_MASK = 0;
int FEET_MASK = 0;
int BODY_COLOR_MASK = 0;
int SIZE_MASK = 0;
int HEAD_COLOR_MASK = 0;

int antennae_start = 0;
int head_start = 4;
int wing_start = 6;
int body_start = 10;
int feet_start = 16;
int body_color_start = 17;
int size_start = 20;
int head_color_start = 22;
int head_color_end = 25;

void setup_masks()
{
	// compute the masks
	for (int i = antennae_start; i < head_start; i++)		SetBit(ANTENNAE_MASK, i);
	for (int i = head_start; i < wing_start; i++)			SetBit(HEAD_MASK, i);
	for (int i = wing_start; i < body_start; i++)			SetBit(WINGS_MASK, i);
	for (int i = body_start; i < feet_start; i++)			SetBit(BODY_MASK, i);
	for (int i = feet_start; i < body_color_start; i++)		SetBit(FEET_MASK, i);
	for (int i = body_color_start; i < size_start; i++)		SetBit(BODY_COLOR_MASK, i);
	for (int i = size_start; i < head_color_start; i++)		SetBit(SIZE_MASK, i);
	for (int i = head_color_start; i < head_color_end; i++) SetBit(HEAD_COLOR_MASK, i);
}


void SetAntennae(int dna, int choice)
{
	dna |= (ANTENNAE_MASK & (choice));
}
void SetHead(int dna, int choice)
{
	dna |= (HEAD_MASK & (choice << head_start));
}
void SetWing(int dna, int choice)
{
	dna |= (WINGS_MASK & (choice << wing_start));
}
void SetBody(int dna, int choice)
{
	dna |= (BODY_MASK & (choice << body_start));
}
void SetFeet(int dna, int choice)
{
	dna |= (FEET_MASK & (choice << feet_start));
}
void SetBodyColor(int dna, int choice)
{
	dna |= (BODY_COLOR_MASK & (choice << body_color_start));
}
void SetSize(int dna, int choice)
{
	dna |= (SIZE_MASK & (choice << size_start));
}
void SetHeadColor(int dna, int choice)
{
	dna |= (HEAD_COLOR_MASK & (choice << head_color_start));
}
int GetAntennae(int dna)
{
	return dna & (ANTENNAE_MASK);
}
int GetHead(int dna)
{
	return ((dna & HEAD_MASK) >> head_start);
}
int GetWing(int dna)
{
	return ((dna & WINGS_MASK) >> wing_start);
}
int GetBody(int dna)
{
	return ((dna & BODY_MASK) >> body_start);
}
int GetFeet(int dna)
{
	return ((dna & FEET_MASK) >> feet_start);
}
int GetBodyColor(int dna)
{
	return ((dna & BODY_COLOR_MASK) >> body_color_start);
}
int GetSize(int dna)
{
	return ((dna & SIZE_MASK) >> size_start);
}
int GetHeadColor(int dna)
{
	return ((dna & HEAD_COLOR_MASK) >> head_color_start);
}


inline float RandomFloat(float min, float max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}


inline float RandomInt(int min, int max)
{
	float r = (float)rand() / (float)RAND_MAX;
	return (int)((float)min + r * float(max - min));
}

void mutate(int dna, float mutationRate) {

	if (RandomFloat(0.0, 1.0) < mutationRate) {
		int32_t gene_to_mutate = RandomInt(0, 25);

		if (dna & (1 << gene_to_mutate))
		{
			dna |= (0 << gene_to_mutate);
		}
		else dna |= (1 << gene_to_mutate);
	}
}

/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numElements.
*/
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

__global__ void
mutate_cuda(const int *A, const float *B, int *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}

	// this test requires global variable mutation_rate
	float mutation_rate = 0.0;
	if (B[i] >= mutation_rate ) {
		int gene_to_mutate = A[i];

		if (C[i] & (1 << gene_to_mutate))
		{
			C[i] |= (0 << gene_to_mutate);
		}
		else C[i] |= (1 << gene_to_mutate);
	}
}

template <class T>
__global__ void
reduce3(T *g_idata, T *g_odata, unsigned int n)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	T *sdata = SharedMemory<T>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;

	T mySum = (i < n) ? g_idata[i] : 0;

	if (i + blockDim.x < n)
		mySum += g_idata[i + blockDim.x];

	sdata[tid] = mySum;
	cg::sync(cta);

	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] = mySum = mySum + sdata[tid + s];
		}

		cg::sync(cta);
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = mySum;
}


void print_dna(int g)
{
	for (int i = 0; i < 32; i++)
	{
		if (CheckBit(g, i)) printf("1");
		else printf("0");
	}
}

/**
*
*/
void TestMutate() {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000;
	size_t float_size = numElements * sizeof(float);
	size_t int_size = numElements * sizeof(int);
	printf("[Test for mutation operation in CUDA of %d elements]\n", numElements);

	// Allocate the host input vector A
	int *h_A = (int *)malloc(int_size);

	// Allocate the host input vector B
	float *h_B = (float *)malloc(float_size);

	// Allocate the host output vector C
	int *h_C = (int *)malloc(int_size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = RandomInt(0, 25);
		h_B[i] = RandomFloat(0, 1);
	}

	// Allocate the device input vector A
	int *d_A = NULL;
	err = cudaMalloc((void **)&d_A, int_size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, float_size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	int *d_C = NULL;
	err = cudaMalloc((void **)&d_C, int_size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, int_size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, float_size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	mutate_cuda << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, int_size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (CheckBit(h_C[i], h_A[i]) == 0)
			//if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
		//else don't print because its a bunch of 1's and 0's
		//{
		//	print_dna(h_C[i]);
		//}
	}

	printf("Test PASSED\n");

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
template<class T>
T reduceCPU(T *data, int size)
{
	T sum = data[0];
	T c = (T)0.0;

	for (int i = 1; i < size; i++)
	{
		T y = data[i] - c;
		T t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}

	return sum;
}

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

	//get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device;
	checkCudaErrors(cudaGetDevice(&device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));

	if (whichKernel < 3)
	{
		threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
		blocks = (n + threads - 1) / threads;
	}
	else
	{
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
	}

	if ((float)threads*blocks >(float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
	{
		printf("n is too large, please choose a smaller number!\n");
	}

	if (blocks > prop.maxGridSize[0])
	{
		printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
			blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	if (whichKernel == 6)
	{
		blocks = MIN(maxBlocks, blocks);
	}
}





////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduce(int size, int threads, int blocks,
	int whichKernel, T *d_idata, T *d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);




		reduce3<T> << < dimGrid, dimBlock, smemSize >> >(d_idata, d_odata, size);



}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int  n,
	int  numThreads,
	int  numBlocks,
	int  maxThreads,
	int  maxBlocks,
	int  whichKernel,
	int  testIterations,
	bool cpuFinalReduction,
	int  cpuFinalThreshold,
	StopWatchInterface *timer,
	T *h_odata,
	T *d_idata,
	T *d_odata)
{
	T gpu_result = 0;
	bool needReadBack = true;

	for (int i = 0; i < testIterations; ++i)
	{
		gpu_result = 0;

		cudaDeviceSynchronize();
		sdkStartTimer(&timer);

		// execute the kernel
		reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

		// check if kernel execution generated an error
		getLastCudaError("Kernel execution failed");
		// Clear d_idata for later use as temporary buffer.
		cudaMemset(d_idata, 0, n * sizeof(T));

		if (cpuFinalReduction)
		{
			// sum partial sums from each block on CPU
			// copy result from device to host
			checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(T), cudaMemcpyDeviceToHost));

			for (int i = 0; i<numBlocks; i++)
			{
				gpu_result += h_odata[i];
			}

			needReadBack = false;
		}
		else
		{
			// sum partial block sums on GPU
			int s = numBlocks;
			int kernel = whichKernel;

			while (s > cpuFinalThreshold)
			{
				int threads = 0, blocks = 0;
				getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);
				cudaMemcpy(d_idata, d_odata, s * sizeof(T), cudaMemcpyDeviceToDevice);
				reduce<T>(s, threads, blocks, kernel, d_idata, d_odata);

				if (kernel < 3)
				{
					s = (s + threads - 1) / threads;
				}
				else
				{
					s = (s + (threads * 2 - 1)) / (threads * 2);
				}
			}

			if (s > 1)
			{
				// copy result from device to host
				checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

				for (int i = 0; i < s; i++)
				{
					gpu_result += h_odata[i];
				}

				needReadBack = false;
			}
		}

		cudaDeviceSynchronize();
		sdkStopTimer(&timer);
	}

	if (needReadBack)
	{
		// copy final sum from device to host
		checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
	}

	return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
/**
*
*/
void TestReduceFitness() {

	int maxThreads = 256;  // number of threads per block
	int whichKernel = 3;
	int maxBlocks = 64;
	bool cpuFinalReduction = false;
	int cpuFinalThreshold = 1;



	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	int numElements = 50000;
	size_t float_size = numElements * sizeof(float);
	size_t int_size = numElements * sizeof(int);

	// create random input data on CPU
	unsigned int bytes = numElements * sizeof(float);

	float *h_idata = (float *)malloc(bytes);

	for (int i = 0; i<numElements; i++)
	{
		// Keep the numbers small so we don't get truncation error in the sum
		{
			h_idata[i] = (rand() & 0xFF) / (float)RAND_MAX;
		}
	}

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, numElements, maxBlocks, maxThreads, numBlocks, numThreads);

	if (numBlocks == 1)
	{
		cpuFinalThreshold = 1;
	}

	// allocate mem for the result on host side
	float *h_odata = (float *)malloc(numBlocks * sizeof(float));

	printf("%d blocks\n\n", numBlocks);

	// allocate device memory and data
	float *d_idata = NULL;
	float *d_odata = NULL;

	checkCudaErrors(cudaMalloc((void **)&d_idata, bytes));
	checkCudaErrors(cudaMalloc((void **)&d_odata, numBlocks * sizeof(float)));

	// copy data directly to device memory
	checkCudaErrors(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_odata, h_idata, numBlocks * sizeof(float), cudaMemcpyHostToDevice));

	// warm-up
	reduce<float>(numElements, numThreads, numBlocks, whichKernel, d_idata, d_odata);

	int testIterations = 100;

	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);

	float gpu_result = 0;

	gpu_result = benchmarkReduce<float>(numElements, numThreads, numBlocks, maxThreads, maxBlocks,
		whichKernel, testIterations, cpuFinalReduction,
		cpuFinalThreshold, timer,
		h_odata, d_idata, d_odata);

	double reduceTime = sdkGetAverageTimerValue(&timer) * 1e-3;
	printf("Reduction, Throughput = %.4f GB/s, Time = %.5f s, Size = %u Elements, NumDevsUsed = %d, Workgroup = %u\n",
		1.0e-9 * ((double)bytes) / reduceTime, reduceTime, numElements, 1, numThreads);

	// compute reference solution
	float cpu_result = reduceCPU<float>(h_idata, numElements);

	int precision = 0;
	double threshold = 0;
	double diff = 0;



			precision = 8;
			threshold = 1e-8 * numElements;

		printf("\nGPU result = %.*f\n", precision, (double)gpu_result);
		printf("CPU result = %.*f\n\n", precision, (double)cpu_result);

		diff = fabs((double)gpu_result - (double)cpu_result);
	

	// cleanup
	sdkDeleteTimer(&timer);
	free(h_idata);
	free(h_odata);

	checkCudaErrors(cudaFree(d_idata));
	checkCudaErrors(cudaFree(d_odata));


	if (diff < threshold)printf("\n diff < threshold");
	
	
	
	

}

/**
* Host main routine
*/
int
main(void)
{
	//TestMutate();
	TestReduceFitness();
	return 0;
}

