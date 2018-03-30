#include <iostream>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

// not used here, but will be later
//#include <cooperative_groups.h>


/**
 * Host main routine
 */
int
insect_test(bool b_verify);

#define WARP_SIZE 32









//// 
//// 	PORTIONS OF THIS CODE ARE BASED ON NVIDIA SAMPLES BUT HIGHLY MODIFIED
//// 	THE PORTIONS THAT CONTAIN NVIDIA CODE ARE EXEMPT FROM THE GPL
////
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




/*
Modern Graphics Processing Units (GPU's) contain massively parallel thread processing
capability. CUDA programs are instructions for the individual threads.

The top most thread structure of a CUDA gpu program is the grid. The grid contains
multiple tiles called blocks, and the blocks contain threads. Grids and blocks can have
up to 3 dimensions of indices.

On modern hardware a typical maximum block size is 1024 threads, however they are often a lot less. 
The threads in each block are launched in warps (sets) of 32 threads. Each thread in a warp executes at the same time 
in the hardware, however due to program branches they also need to be synchronized. The program can achieve block level
synchronization with a call to "__syncthreads()", however on newer hardware with larger blocks some alternatives
to block level synchronisation have appeared, including: cooperative groups; "__syncwarp()", a warp level synchronisation call;
and functions like "__shfl_down_sync()" which perform warp level shuffle and synchronisation operations. 

The memory structure on CUDA GPU's is simple. Each thread has "slow" access to the larger global memory, each block has a small shared
memory (faster) and register memory space (fastest). Array's of data uploaded from the CPU (host) RAM to the GPU (device) VRAM will typically reside in
global memory. Shared memory is declared in the CUDA program "kernel" using the "__shared__" keyword, and data must be fetched from global memory (or texture)
into the shared memory, or generated in kernel.bThe shared memory can be very useful for warp level processing.   

Terminology:
blockIndx : This is the index of the block within the grid.
blockDim : The number of threads per block
threadIdx : The index of the thread in the block
warpID : the identifier of the warp in the block
laneID : the index of the thread within the warp.


This program takes advantage of warp level synchronization (or will do) to compute genetic algorithms (GA) where each warp represents a population (gene pool).
 



*/


// stack overflow
__forceinline__ __device__ unsigned int get_lane_id() // //threadIdx.x % WARP_SIZE;
{
	unsigned int x;
	asm volatile("mov.u32 %0, %laneid;":"=r"(x));
	return x;
}

__forceinline__ __device__ unsigned int get_warp_id() //threadIdx.x / WARP_SIZE;
{
	unsigned int x;
	asm volatile("mov.u32 %0, %warpid;":"=r"(x));
	return x;
}


// CUB
__device__ __forceinline__ int bfe(int val, int lane)
{
	unsigned x;
	asm volatile("bfe.u32 %0, %1, %2, 1;" : "=r"(x) : "r"(val), "r"(lane));
	return x;
}


template<typename T>
__device__ 
T warp_readFromLane(T x, int lID)
{
	return __shfl_sync(0xffffffff, x, lID);
}





// bitfield
 int SetBit( int dna,  int idx)
{
	dna |= (1<<idx);
	return dna;
}

__forceinline__ __host__ __device__
bool CheckBit( int dna,  int idx)
{
	return dna & (1<<idx);
}


bool CheckBit2( int dna,  int idx)
{
	return dna & (1<<idx);
}


 int ResetBit( int dna,  int idx)
{
	dna &= ~(1<<idx);
	return dna;
}


struct Insect
{
	 int dna;
	float fitness;
	float sumFitness;
	//unsigned int cross_selection;

	__device__ void operator=(Insect &o){
		this->dna = o.dna;
		this->fitness = o.fitness;
		//this->cross_selection = o.cross_selection;
	}
};



__device__ 
void warp_insect_all_reduce(Insect shared[WARP_SIZE],int laneID)
{

	#pragma unroll
	for(unsigned int offset = WARP_SIZE/2; offset > 0; offset >>=1)
	{
		__syncwarp();
		if( laneID < offset ) shared[laneID].fitness+=shared[laneID+offset].fitness;
	}
}

__device__ 
void warp_insect_all_reduce_2(Insect shared[WARP_SIZE],int laneID)
{
	shared[laneID].sumFitness =shared[laneID].fitness;

	__syncwarp();
	#pragma unroll
	for(unsigned int offset = WARP_SIZE/2; offset > 0; offset >>=1)
	{
		__syncwarp();
		if( laneID < offset ) shared[laneID].sumFitness += shared[laneID+offset].sumFitness;
	}
}

// Kepler's shuffle tips and tricks
// broken
__device__ 
unsigned int warp_insect_scan(Insect mem[WARP_SIZE], unsigned int i)
{
	mem[i].sumFitness =mem[i].fitness;
	__syncwarp();
	#pragma unroll
	for( int offset = 1; offset < WARP_SIZE; offset <<= 1 )
	{
		float y = __shfl_up_sync(0xffffffff, mem[i].sumFitness, offset);
		if(  i >= offset )
		{
			mem[i].sumFitness += y;
			//i += yidx;
		}
	}

	return i;
	
}





__device__ 
unsigned int swap_insect(Insect mem[32], unsigned int i, int mask, int dir)
{
	unsigned int yidx = __shfl_xor_sync(0xffffffff,i,mask);
	Insect y = mem[yidx];
	return (mem[i].fitness < y.fitness)==dir ? yidx : i;
}

// Kepler's shuffle tips and tricks
/* Modified by David Nash, all rights reserved */
__device__ 
unsigned int warp_bitonic_sort_insect(unsigned int x, Insect mem[32])
{
	const int laneId = get_lane_id();

	x = swap_insect(mem, x, 0x01, bfe(laneId, 1) ^ bfe(laneId,0));
	x = swap_insect(mem, x, 0x02, bfe(laneId, 2) ^ bfe(laneId,1));
	x = swap_insect(mem, x, 0x01, bfe(laneId, 2) ^ bfe(laneId,0));
	x = swap_insect(mem, x, 0x04, bfe(laneId, 3) ^ bfe(laneId,2));
	x = swap_insect(mem, x, 0x02, bfe(laneId, 3) ^ bfe(laneId,1));
	x = swap_insect(mem, x, 0x01, bfe(laneId, 3) ^ bfe(laneId,0));
	x = swap_insect(mem, x, 0x08, bfe(laneId, 4) ^ bfe(laneId,3));
	x = swap_insect(mem, x, 0x04, bfe(laneId, 4) ^ bfe(laneId,2));
	x = swap_insect(mem, x, 0x02, bfe(laneId, 4) ^ bfe(laneId,1));
	x = swap_insect(mem, x, 0x01, bfe(laneId, 4) ^ bfe(laneId,0));
	x = swap_insect(mem, x, 0x10, 		   bfe(laneId,4));
	x = swap_insect(mem, x, 0x08,  		   bfe(laneId,3));
	x = swap_insect(mem, x, 0x04,  		   bfe(laneId,2));
	x = swap_insect(mem, x, 0x02,  		   bfe(laneId,1));
	x = swap_insect(mem, x, 0x01,  		   bfe(laneId,0));

	return x;
}


void print_dna(int g)
{
	for( int i = 0 ; i < 32; i++)
	{
		if(CheckBit2(g,i))printf("1");
		else printf("0");
	}
}




// RNG init kernel
// This is found in the Nvidia development guide
__global__ void initRNG(curandState *const rngStates,
                        float *d_B)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG d_B[tid]*4
    curand_init(4, tid, 0, &rngStates[tid]);
}



/*
Data collected for each warp
This data is in the form 
numThreads / WARP_SIZE
*/
struct warp_data
{
	//float warp_mean;
	float warp_reduced;
	Insect warp_best;
	int complete;
	int dna;

	unsigned int blockIdx;
	unsigned int warpId;
	
	__device__ void operator=(warp_data &o)
	{
		this->warp_reduced = o.warp_reduced;
		this->warp_best = o.warp_best;
		this->complete = o.complete;
		this->dna = o.dna;
		this->blockIdx = o.blockIdx;
		this->warpId = o.warpId;

	}
};

struct ret_data
{
bool complete;
unsigned int warp_id;
unsigned int thread_id;
 int dna;
};



/*
copyright: David Nash, all rights reserved
*/
__global__ void run_GA(Insect *in, int target, Insect *out, curandState *const rngStates, int N, warp_data* wd,  int num_warps)
{
	
	//curandState local_state;
	//local_state = global_state[threadIdx.x];
	// cuDa ve
	int bi = blockIdx.x*blockDim.x;

	static __shared__ 
	Insect shared[WARP_SIZE*2];
	static __shared__	
	Insect swap[WARP_SIZE*2];

	static __shared__ warp_data _warp_data;
	
	
	
	
	unsigned int laneID = get_lane_id();
	unsigned int warpID = get_warp_id();

	unsigned int tid = bi+threadIdx.x;

	unsigned int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE;	

	    // Initialise the RNG
    	curandState localState = rngStates[tid];
	
	float randomFloat = curand_uniform(&localState);
	
	in[tid].fitness = 0.0f;
	in[tid].sumFitness = 0.0f;


	// load global data to shared

	shared[laneID] = in[tid];
	_warp_data = wd[ warpID];

	_warp_data.blockIdx = blockIdx.x;
	_warp_data.warpId = warpID;
	

	/* Migrations */
	/*if( (warpID > 0)  && laneID == 0 )
		swap[laneID] = in[tid-1];*/

	
	// mutation
	float mutationRate = 0.01 *(1 + warpID) ; // this seems to make later warps perform better
	int dna =  shared[laneID].dna;
	if( randomFloat < mutationRate )
	{
		int gene_to_mutate=(int)(curand_uniform(&localState)*25);

		if( dna &(1<<gene_to_mutate))
		{
			dna  |= (0<< gene_to_mutate);
		}
		else {
			dna |= (1<<gene_to_mutate);
		}
		// BAD IDEA
		//print_dna( out[tid].dna);

	}
	shared[laneID].dna = dna;
	__syncwarp();
	

	// fitness score
	// This could be performed using a separate kernel launch
	// with threads bishifting loaded data by laneID (position in the warp)
	// followed by a comparison to target (or evaluation function) 
	unsigned int count = 0;//{};
	//count[laneID] = 0;
	
	
	for( int i = 0; i < 25; i++ )
	{
		if( CheckBit(shared[laneID].dna,i) )
		{
			if( CheckBit(target, i))
			{
				count++;
			}
		}
		else
		{
			if( !CheckBit(target, i))
			{
				count++;
			}
		}
	}
	__syncwarp();

	// compute fitness based on the accumulated samples
	if( count == 0 )
	{ 
		shared[laneID].fitness = 0.05f;
	}
	else
	{
		if( count == 25 ) {
			//printf("Finished in loop iteratons, block: %d, warp: %d, thread: %d", blockIdx.x, warpID, laneID);
			shared[laneID].fitness = 10000000;

			_warp_data.complete = 1010101010;
			_warp_data.dna = shared[laneID].dna;
		
		}
		else
		{
			shared[laneID].fitness = ((float)count)/(25.0f);
		}
	}

	__syncwarp();

	// sort per warp
	int newID = warp_bitonic_sort_insect(laneID, shared);

	swap[laneID] = shared[newID];

	__syncwarp();

	_warp_data.warp_best = swap[WARP_SIZE-1];
	
	// perform scan operation for roulette wheel mating selection
	warp_insect_scan(swap, laneID);

	// store this in local memory now, then write to global memory at end of kernel
	Insect this_insect = swap[laneID];

	

	__syncwarp();
	//if( laneID == 0 ) // power saving
	float warp_Reduced = _warp_data.warp_reduced = swap[WARP_SIZE-1].sumFitness;

	
	// breeding

	int FULLMASK = 0xffffffff;
	int NEWMASK = FULLMASK << 24;


	 
	// can perform a different test for each warp.
	if( warpID % WARPS_PER_BLOCK < WARP_SIZE / 2) // currently competitive with roulette wheel
	{
		/* Attempt to mask out the insects with low fitness 
		 while cloning  
		
		this should copy the top 8 insects over the rest by selecting memory at offset laneID % 8 + 24 
		
		*/
		float breedingSelector = curand_uniform(&localState) * 31;


		int crossover_selection = (int)breedingSelector;

		swap[laneID].dna = __shfl_sync(~NEWMASK, swap[laneID].dna, (laneID % 8 + 24));

		//crossover_selection = __shfl_sync(NEWMASK, laneID-1, (laneID % 8 + 24));
	
		// cloning the best index
		NEWMASK = FULLMASK << 16;
		int tempDNA = (swap[laneID].dna & (NEWMASK)) | (swap[crossover_selection].dna & (~NEWMASK));
		
		__syncwarp();

		swap[laneID].dna = tempDNA;
		//swap[laneID].dna = shared[laneID].dna;
	}
	else
	{

		float breedingSelector = curand_uniform(&localState) * warp_Reduced;
		breedingSelector *= breedingSelector;


		int crossover_selection = (int)breedingSelector;

		//NEWMASK = FULLMASK << 8; // swap at 50%

		//swap[laneID].dna = __shfl_sync(~NEWMASK, swap[laneID].dna, (laneID % 8 + 24));

		int TESTER = WARP_SIZE / 2;
		
		
		
		// this is supposed to be a binary search over the 32 values to find the tcorrec
		// 		 inddex for the breeding 
		
		// Because the cumulative sum of fitness is stored in each consecutive sorted member 
		// we should be able to find the member whose sumfitness is less than randSelector and whose next neighbour
		// has sumfitness greater ... using binary search roullette wheel
		
		for( int INCREMENT = TESTER/2; INCREMENT > 0; INCREMENT /=2 )
		{
			if( swap[TESTER -1].sumFitness*swap[TESTER -1].sumFitness >  breedingSelector ) TESTER -= INCREMENT;
			else if( swap[TESTER].sumFitness*swap[TESTER].sumFitness <=  breedingSelector ) TESTER += INCREMENT;
		}
		crossover_selection = TESTER; 
		
		NEWMASK = FULLMASK << 16; // swap at 50%
		int tempDNA = (swap[laneID].dna & (NEWMASK)) | (swap[crossover_selection].dna & (~NEWMASK));
		
		__syncwarp();

		swap[laneID].dna = tempDNA;
	}

	wd[ warpID] = _warp_data;
	
	


		/*migrations*/
	if( laneID == 0 && warpID % WARPS_PER_BLOCK > 0  )
		swap[0] = wd[warpID-1].warp_best;

	__syncthreads();

	
	out[tid] = this_insect;

	in[tid].dna = swap[laneID].dna;
	in[tid].fitness = 0.0f;
	in[tid].sumFitness = 0.0f;

	
	
	


	// thread migration
	//if( warpID-1 > -1 && laneID == 0 )
	//	out[warpID-1] = out[warpID];

	//__syncthreads();

	// WARP (laneID + 1) % 32; this reads the next lanes memory, but %32 is the remainder of division by 32 so when laneID=31 we get 32%32=0, and 33%32=1 etc, the last lane reads from the first

	// cloning

	//swap[0] =swap[WARP_SIZE-1];

	// store per warp data
	


	
	
	//in[tid] = out[tid];

	//__syncwarp();

	//in[tid]dna = 0;


}

template< typename T >
T* allocate_device(T* d_A, int N)
{
	int size = sizeof(T)*N;
	//
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

	return d_A;
}


void memcpy_host_to_device(void* host, void* dev, size_t size)
{
    printf("Copy input data from the host memory to the CUDA device\n");
    cudaError_t err = cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// cuda toolkit documentation
// must fix this because hostData is still allocated here
/*float* cuda_random_generate_array(int N)
{
	curandStatus_t err = CURAND_STATUS_SUCCESS;
	curandGenerator_t gen;
	float *devData, *hostData;
	hostData = (float*)calloc(N, sizeof(float));
	devData = allocate_device<float>(devData,N);
	err = curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	if( err != CURAND_STATUS_SUCCESS )
	{
		printf("Failed to create random number generator!");
		exit(EXIT_FAILURE);
	}
	err = curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
	if( err != CURAND_STATUS_SUCCESS )
	{
		printf("Failed to set random number generator seed!");
		exit(EXIT_FAILURE);
	}	
	err = curandGenerateUniform(gen,devData,N);
	if( err != CURAND_STATUS_SUCCESS )
	{
		printf("Failed to generate uniform random numbers!");
		exit(EXIT_FAILURE);
	}
	return devData;
}*/




int main(void)
{

int FULLMASK = 0xffffffff;

int test = FULLMASK << 24;
	print_dna(~test);
    insect_test(true);

  return 0;
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

void ComputeMasks()
{
	for( int i = antennae_start; i < head_start; i++) ANTENNAE_MASK |=SetBit(ANTENNAE_MASK,i);
	for( int i = head_start; i < wing_start; i++) HEAD_MASK|=SetBit(HEAD_MASK,i);
	for( int i = wing_start; i < body_start; i++) WINGS_MASK|=SetBit(WINGS_MASK,i);
	for( int i = body_start; i < feet_start; i++) BODY_MASK|=SetBit(BODY_MASK,i);
	for( int i = feet_start; i < body_color_start; i++) FEET_MASK|=SetBit(FEET_MASK,i);
	for( int i = body_color_start; i < size_start; i++) BODY_COLOR_MASK|=SetBit(BODY_COLOR_MASK,i);
	for( int i = size_start; i < head_color_start; i++) SIZE_MASK|=SetBit(SIZE_MASK,i);
	for( int i = head_color_start; i < head_color_end; i++) HEAD_COLOR_MASK|=SetBit(HEAD_COLOR_MASK,i);
}

int * SetAntennae( int *dna,  int choice)
{
	*dna |= (ANTENNAE_MASK &(choice) ) ;
	return dna;
}
int * SetHead( int *dna, int choice)
{
	*dna |= (HEAD_MASK &(choice << head_start));
	return dna;
}
int * SetWings( int *dna,  int choice)
{
	*dna |= (WINGS_MASK &(choice << wing_start));
	return dna;
}
 int * SetBody( int *dna,  int choice)
{
	*dna |= (BODY_MASK &(choice << body_start));
	return dna;
}
 int * SetFeet( int *dna,  int choice)
{
	*dna |= (FEET_MASK &(choice << feet_start));
	return dna;
}
 int * SetBodyColor( int *dna, int choice)
{
	*dna |= (BODY_COLOR_MASK &(choice << body_color_start));
	return dna;
}
int * SetSize( int *dna, int choice)
{
	*dna |= (SIZE_MASK &(choice << size_start));
	return dna;
}
 int * SetHeadColor( int *dna,  int choice)
{
	*dna |= (HEAD_COLOR_MASK &(choice << head_color_start));
	return dna;
}

void SetInsect(int* dna)
{
	*dna = *SetAntennae( dna, 2);
	print_dna(*dna);
	printf(", 0x%08x\n", dna);
	 *dna = *SetHead(dna, 4);
	print_dna(*dna);printf(", 0x%08x \n", dna);
	 *dna = *SetWings( dna,34);
	print_dna(*dna);printf(", 0x%08x\n", dna);
	 *dna = *SetBody( dna, 5);
	print_dna(*dna);printf(", 0x%08x\n", dna);
	 *dna = *SetFeet( dna, 1);
	print_dna(*dna);printf(", 0x%08x\n", dna);
	 *dna = *SetBodyColor( dna,3);
	print_dna(*dna);printf(", 0x%08x\n", dna);
	 *dna = *SetSize( dna, 2);
	print_dna(*dna);printf(", 0x%08x\n", dna);
	*dna = *SetHeadColor( dna, 1);	
	print_dna(*dna);printf(", 0x%08x\n", dna);printf("\n");
	//return dna;
}

/**
 * Host main routine
 */
int
insect_test(bool b_verify)
{

ComputeMasks();
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;



    // Print the vector length to be used, and compute its size
	
    int numElements = 32768;
  	int threadsPerBlock = 128; // optimized for SM running 128 threads or 4 warps simultaneously
	int warps_per_block = threadsPerBlock / 32;
    	int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    size_t size = numElements * sizeof(float);
	size_t size_int = numElements * sizeof(int);

size_t insect_size = sizeof(Insect)*numElements;
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    Insect *h_A = (Insect *)malloc(insect_size);
    float *h_B = (float *)malloc(size);
    Insect *h_C = (Insect *)malloc(insect_size);
	warp_data *wdH = (warp_data*)malloc( numElements * sizeof(warp_data) / 32) ;

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }



    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i].dna = (rand()/(float)RAND_MAX)*320;
	h_A[i].fitness = 0.0f;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    Insect *d_A = NULL;
	d_A = allocate_device<Insect>(d_A, numElements);
    float *d_B = NULL;
	d_B = allocate_device<float>(d_B, numElements);
    // Allocate the device output vector C
    Insect *d_C = NULL;
	d_C = allocate_device<Insect>(d_C, numElements);

	warp_data *D_wd = NULL;//(warp_data*)malloc( numElements * sizeof(warp_data) / 32) ;
	  D_wd = allocate_device<warp_data>(D_wd, numElements * sizeof(warp_data) / 32);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
	//memcpy_host_to_device(h_A, d_A, size);
    	printf("Copy input data from the host memory to the CUDA device\n");
    	err = cudaMemcpy(d_A, h_A, insect_size, cudaMemcpyHostToDevice);

    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}

    	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	//memcpy_host_to_device(h_B, d_B, size);
    	if (err != cudaSuccess)
    	{
        	fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
    	}


	// Launch the Vector Add CUDA Kernel
  
	//int blocksPerGrid =(numElements ) / threadsPerBlock;

	struct cudaFuncAttributes funcAttributes;

    	// Get initRNG function properties and check the maximum block size
    	err = cudaFuncGetAttributes(&funcAttributes, initRNG);

    	if (err != cudaSuccess)
    	{
        	printf("Could not get function attributes: ");
		exit(EXIT_FAILURE);        
		//msg += cudaGetErrorString(cudaResult);
        	//throw std::runtime_error(msg);
    	}

    	if (threadsPerBlock > (unsigned int)funcAttributes.maxThreadsPerBlock)
    	{
        	printf("Block X dimension is too large for initRNG kernel");
		exit(EXIT_FAILURE); 
    	}

   // Allocate memory for RNG states
    	curandState *d_rngStates = 0;
    	err = cudaMalloc((void **)&d_rngStates, blocksPerGrid * threadsPerBlock * sizeof(curandState));

    	if (err != cudaSuccess)
    	{
        	printf("Could not allocate memory on device for RNG state: ");
        	exit(EXIT_FAILURE); 
    	}





    	// Initialise RNG
    	initRNG<<<blocksPerGrid, threadsPerBlock>>>(d_rngStates, d_B);

	 int insect_dna = 0;

	
	SetInsect(&insect_dna);




	cudaEvent_t start,stop;
    		cudaEventCreate(&start);
    		cudaEventCreate(&stop);

    		cudaEventRecord(start);
    
	int p = 0;
    	for( p = 0; p < 30; p++ )
	{
	
    		//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


		

		run_GA<<<blocksPerGrid,threadsPerBlock>>>(d_A, insect_dna, d_C, d_rngStates, numElements, D_wd, numElements/32);

	
		
		//printf("Copy output data from the CUDA device to the host memory\n");
    		err = cudaMemcpy(wdH, D_wd, sizeof(warp_data)*numElements / 32, cudaMemcpyDeviceToHost);
    		if (err != cudaSuccess)
    		{
        		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        		exit(EXIT_FAILURE);
    		}
    		

	

		bool target_reached = false;

		for( int i = 0; i < numElements / 32; i++ )
		{
			if( wdH[i].complete == 1010101010 )
			{
				target_reached = true;
				printf("TARGET_REACHED\n------------------------------\n");
				printf("found at block ; %d, warp: %d \n\n", wdH[i].warp_reduced, wdH[i].warp_best.fitness, wdH[i].blockIdx, wdH[i].warpId % warps_per_block );
				printf("Insect: ");
				print_dna(wdH[i].dna);
				printf(" \nTarget: ");
				print_dna(insect_dna);
				break;
			}
			
		}

		if( target_reached )


		{
			printf("\n\n Target Reached at iteration %d\n", p);
			break;
		}
	}

		cudaEventRecord(stop);

	    	cudaEventSynchronize(stop);

    		float milliseconds = 0;
    		cudaEventElapsedTime(&milliseconds, start, stop);

		// Mark Harris tutoral

    		/*// size in bytes * 3 memory accesses, 2 retrieve, one store (unit converted)
    		printf("Effective Bandwidth (GB/s): %f\n", insect_size * 32 / milliseconds / 1e6 );  

    		// one floating point operation (add) * numElements (unit converted)
    		printf("Throughput (GFLOPS): %f\n", 5*numElements / milliseconds / 1e6 );  
*/
    		printf( "Time Elapsed (ms): %f\n", milliseconds);
		
    		err = cudaGetLastError();

	printf("\ncompleted %d iterations \n", p);


	    if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }

	    // Copy the device result vector in device memory to the host result vector
	    // in host memory.
	    printf("Copy output data from the CUDA device to the host memory, this should be stored in a texture for larger experiements\n");
	    err = cudaMemcpy(h_C, d_C, insect_size, cudaMemcpyDeviceToHost);
	    if (err != cudaSuccess)
	    {
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	    }

	   
		// debug information
	    if( false )
	    {
		// Verify that the result vector is correct
		for (int i = 1; i < numElements; ++i)
		{
	 
			//printf("%f, ", (double)h_C[i].sumFitness);
		}

		for( int i = 0; i < numElements / 32; i++ )
		{
			if( i%64 ==0)
			{	printf("warp_reduced: %f, warp_best: %f, \n", wdH[i].warp_reduced, wdH[i].warp_best.fitness);
					print_dna(wdH[i].warp_best.dna);
					printf(" \n");
					print_dna(insect_dna);printf(" \n");printf(" \n");
			}
		}
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
    if (d_rngStates)
    {
        cudaFree(d_rngStates);
        d_rngStates = 0;
    }


    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

