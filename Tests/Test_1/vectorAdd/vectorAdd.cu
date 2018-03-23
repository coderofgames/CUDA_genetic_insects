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
	if (B[i] >= mutation_rate) {
		int gene_to_mutate = A[i];

		if (C[i] & (1 << gene_to_mutate))
		{
			C[i] |= (0 << gene_to_mutate);
		}
		else C[i] |= (1 << gene_to_mutate);
	}
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
* Host main routine
*/
int
main(void)
{
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
		h_A[i] = RandomInt(0,25);
		h_B[i] = RandomFloat(0,1);
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
	mutate_cuda << <blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, numElements);
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
		if( CheckBit(h_C[i], h_A[i])==0 )
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
	return 0;
}

