#include <iostream>
#include <math.h>
#include <curand.h>



/**
 * Host main routine
 */
int
insect_test(bool b_verify);

// function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}




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

#include <cooperative_groups.h>

//#include <helper_cuda.h>
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



template<typename T>
__device__ 
T swap(T x, int mask, int dir)
{
	T y = __shfl_xor_sync(0xffffffff, x,mask);
	return (x < y)==dir ? y : x;
}


__device__ __forceinline__ int bfe(int val, int lane)
{
	unsigned x;
	asm volatile("bfe.u32 %0, %1, %2, 1;" : "=r"(x) : "r"(val), "r"(lane));
	return x;
}

// not this
//#define bfe(i,k) i &(1 << k)

// Kepler's shuffle tips and tricks
template<typename T>
__device__ 
T warp_bitonic_sort(T x)
{
	const int laneId = get_lane_id();

	x = swap<T>(x, 0x01, bfe(laneId, 1) ^ bfe(laneId,0));
	x = swap<T>(x, 0x02, bfe(laneId, 2) ^ bfe(laneId,1));
	x = swap<T>(x, 0x01, bfe(laneId, 2) ^ bfe(laneId,0));
	x = swap<T>(x, 0x04, bfe(laneId, 3) ^ bfe(laneId,2));
	x = swap<T>(x, 0x02, bfe(laneId, 3) ^ bfe(laneId,1));
	x = swap<T>(x, 0x01, bfe(laneId, 3) ^ bfe(laneId,0));
	x = swap<T>(x, 0x08, bfe(laneId, 4) ^ bfe(laneId,3));
	x = swap<T>(x, 0x04, bfe(laneId, 4) ^ bfe(laneId,2));
	x = swap<T>(x, 0x02, bfe(laneId, 4) ^ bfe(laneId,1));
	x = swap<T>(x, 0x01, bfe(laneId, 4) ^ bfe(laneId,0));
	x = swap<T>(x, 0x10, 		   bfe(laneId,4));
	x = swap<T>(x, 0x08,  		   bfe(laneId,3));
	x = swap<T>(x, 0x04,  		   bfe(laneId,2));
	x = swap<T>(x, 0x02,  		   bfe(laneId,1));
	x = swap<T>(x, 0x01,  		   bfe(laneId,0));

	return x;
}

#define WARP_SIZE 32 

template<typename T>
__device__ 
T warp_readFromLane(T x, int lID)
{
	return __shfl_sync(0xffffffff, x, lID);
}

__inline__ __device__
int fake_shfl_down(int val, int offset, int width=32)
{
	static __shared__ int shared[512];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x]=val;
	__syncthreads();
	val = (lane+offset<width)?shared[threadIdx.x + offset]:0;
	__syncthreads();
	return val;
}

//#define __shfl_down fake_shfl_down

__inline__ __device__
int fake_shfl_up(int val, int offset, int width=32)
{
	static __shared__ int shared[512];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x]=val;
	__syncthreads();
	val = (lane-offset>0)?shared[threadIdx.x - offset]:0;
	__syncthreads();
	return val;
}

//#define __shfl_up fake_shfl_up

/*__inline__ __device__
int fake_shfl_down(int val, int offset, int width=32)
{
	static __shared__ int shared[512];
	int lane = threadIdx.x % 32;
	shared[threadIdx.x]=val;
	__syncthreads();
	val = (lane+offset<width)?shared[threadIdx.x + offset]:0;
	__syncthreads();
	return val;
}*/

//#define __shfl_down fake_shfl_down

template<typename T>
__device__ 
T warp_reduce(T x)
{

	#pragma unroll
	for( int offset = WARP_SIZE/2; offset > 0; offset >>= 1 )
		x+= __shfl_down_sync(0xffffffff, x, offset);

	return x;
	
}



template<typename T>
__device__ 
T warp_all_reduce(T x)
{

	#pragma unroll
	for( int offset = WARP_SIZE/2; offset > 0; offset >>= 1 )
		x+= __shfl_xor_sync(0xffffffff, x, offset);

	return x;
}

// Kepler's shuffle tips and tricks
template<typename T>
__device__ 
T warp_scan(T x)
{
	unsigned int laneID = get_lane_id();
	#pragma unroll
	for( int offset = 0; offset < WARP_SIZE; offset <<= 1 )
	{
		T y = __shfl_up_sync(0xffffffff, x, offset);
		if(  laneID >= offset )
			x+= y;
	}

	return x;
	
}


// devblogs.nvidia.com/faster-parallel-reductions-kepler
template<typename T>
inline __device__ 
T block_reduce(T x)
{
	static __shared__ T shared[WARP_SIZE];
	
	unsigned int laneID = get_lane_id();//threadIdx.x % WARP_SIZE;
	unsigned int warpID = get_warp_id();//threadIdx.x / WARP_SIZE;

	x = warp_all_reduce<T>(x);

	if( laneID == 0 )
		shared[warpID] = x; 

	__syncthreads();
	
	x = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[laneID] : 0;

	if( warpID == 0)
		x = warp_all_reduce<T>(x); // reduction within first warp

	return x;	
}



// devblogs.nvidia.com/faster-parallel-reductions-kepler
template<typename T>
inline __device__ 
T block_scan(T x)
{
	static __shared__ T shared[WARP_SIZE];
	
	unsigned int laneID = get_lane_id();//threadIdx.x % WARP_SIZE;
	unsigned int warpID = get_warp_id();//threadIdx.x / WARP_SIZE;

	x = warp_all_reduce<T>(x);



	__syncthreads();
	


	return x;
	
}


__global__ void device_reduce_block_atomic(float *in, float *out, int N)
{
	float sum = 0.0;
	for(int i=blockIdx.x* blockDim.x + threadIdx.x; i < N; i+=blockDim.x*gridDim.x)
	{
	sum += in[i];
	}
	sum = block_reduce<float>(sum);
	//if(threadIdx.x==0)
	//	atomicAdd(out,sum);
	if( blockIdx.x* blockDim.x + threadIdx.x < N )
		out[blockIdx.x* blockDim.x + threadIdx.x]=sum;
}

/* Pasted from CUDA samples */
////////////////////////////////////////////////////////////////////////////////
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
////////////////////////////////////////////////////////////////////////////////
template< typename T >
__device__ void selection_sort( T *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        T min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            T val_j = data[j];

            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}
/* end Pasted from CUDA samples */


/*
copyright: David Nash, all rights reserved
*/
__global__ void device_sort_warp(float *in, float *out, int N)
{
	// cuDa ve
	int bi = blockIdx.x*blockDim.x;

	static __shared__ float shared[WARP_SIZE];
	
	unsigned int laneID = get_lane_id();
	unsigned int warpID = get_warp_id();

	
	for( int i = 0; i < WARP_SIZE; i++)
	{
		shared[laneID] = in[bi+warpID *WARP_SIZE + laneID];
	}
	__syncthreads();

	//int left = warpID * WARP_SIZE;
	//int right = warpID* WARP_SIZE + WARP_SIZE;
	
	selection_sort<float>(shared, 0, WARP_SIZE-1);

	for( int i = 0; i < WARP_SIZE; i++)
	{
		out[bi+warpID *WARP_SIZE + laneID] =shared[laneID];
	}
	__syncthreads();	
}

/*
copyright: David Nash, all rights reserved
*/
__global__ void device_sort_warp_2(float *in, float *out, int N)
{
	// cuDa ve
	int bi = blockIdx.x*blockDim.x;

	static __shared__ float shared[WARP_SIZE];
	
	unsigned int laneID = get_lane_id();
	unsigned int warpID = get_warp_id();

	
	for( int i = 0; i < WARP_SIZE; i++)
	{
		shared[laneID] = in[bi+warpID *WARP_SIZE + laneID];
	}
	__syncthreads();

	//int left = warpID * WARP_SIZE;
	//int right = warpID* WARP_SIZE + WARP_SIZE;
	
	//selection_sort<float>(shared, 0, WARP_SIZE-1);
	shared[laneID] = warp_bitonic_sort<float>(shared[laneID]);

	for( int i = 0; i < WARP_SIZE; i++)
	{
		out[bi+warpID *WARP_SIZE + laneID] =shared[laneID];
	}
	__syncthreads();	
}

/*
copyright: David Nash, all rights reserved
*/
__global__ void device_sort_warp_3(float *in, float *out, int N)
{
	// cuDa ve
	int bi = blockIdx.x*blockDim.x;

	//static __shared__ float shared[WARP_SIZE];
	
	unsigned int laneID = get_lane_id();
	unsigned int warpID = get_warp_id();

	// much faster
	out[bi+warpID *WARP_SIZE + laneID] = warp_bitonic_sort<float>(in[bi+warpID *WARP_SIZE + laneID]);


}

__forceinline__ __host__ __device__
void SetBit(unsigned int &dna, unsigned int idx)
{
	dna |= (1<<idx);
}

__forceinline__ __host__ __device__
bool CheckBit(unsigned int dna, unsigned int idx)
{
	return dna & (1<<idx);
}

__forceinline__ __host__ __device__
void ResetBit(unsigned int &dna, unsigned int idx)
{
	dna &= ~(1<<idx);
}


struct Insect
{
	unsigned int dna;
	float fitness;
	//unsigned int cross_selection;

	__device__ void operator=(Insect &o){
		this->dna = o.dna;
		this->fitness = o.fitness;
		//this->cross_selection = o.cross_selection;
	}
};

/* Pasted from CUDA samples */
////////////////////////////////////////////////////////////////////////////////
// Modified to sort Insect's

__device__ void selection_sort_insect( Insect *data, int left, int right)
{
    for (int i = left ; i <= right ; ++i)
    {
        Insect min_val = data[i];
        int min_idx = i;

        // Find the smallest value in the range [left, right].
        for (int j = i+1 ; j <= right ; ++j)
        {
            Insect val_j = data[j];

            if (val_j.fitness < min_val.fitness)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        // Swap the values.
        if (i != min_idx)
        {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}



__device__ 
void warp_insect_all_reduce(Insect shared[WARP_SIZE],int laneID)
{

	#pragma unroll
	for(unsigned int offset = WARP_SIZE/2; offset > 0; offset >>=1)
	{
		__syncthreads();
		if( laneID < offset ) shared[laneID].fitness+=shared[laneID+offset].fitness;
	}
}

// Kepler's shuffle tips and tricks
// broken
__device__ 
unsigned int warp_insect_scan(Insect mem[WARP_SIZE], unsigned int i)
{
	//#pragma unroll
	for( int offset = 1; offset < WARP_SIZE; offset <<= 1 )
	{
		float y = __shfl_up_sync(0xffffffff, mem[i].fitness, offset);
		if(  i >= offset )
		{
			mem[i].fitness += y;
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


void print_dna(unsigned int g)
{
	for( unsigned int i = 0 ; i < 32; i++)
	{
		if(CheckBit(g,i))printf("1");
		else printf("0");
	}
}

#include <curand_kernel.h>


// RNG init kernel
__global__ void initRNG(curandState *const rngStates,
                        float *d_B)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG d_B[tid]*4
    curand_init(4, tid, 0, &rngStates[tid]);
}
/* end Pasted from CUDA samples */


struct warp_data
{
	//float warp_mean;
	float warp_reduced;
	Insect warp_best;
	int complete;
	int dna;
};

struct ret_data
{
bool complete;
unsigned int warp_id;
unsigned int thread_id;
unsigned int dna;
};



/*
copyright: David Nash, all rights reserved
*/
__global__ void fitness_samples(Insect *in, unsigned int target, Insect *out, curandState *const rngStates, int N, warp_data* wd,  int num_warps)
{
	//curandState local_state;
	//local_state = global_state[threadIdx.x];
	// cuDa ve
	int bi = blockIdx.x*blockDim.x;

	static __shared__ 
	Insect shared[WARP_SIZE];
	static __shared__	
	Insect swap[WARP_SIZE];
	
	
	
	
	unsigned int laneID = get_lane_id();
	unsigned int warpID = get_warp_id();

	unsigned int tid = bi+threadIdx.x;

	

	    // Initialise the RNG
    	curandState localState = rngStates[tid];
	
	float randomFloat = curand_uniform(&localState);

	shared[laneID] = in[tid];
	
	// mutation
	float mutationRate = 0.5;
	unsigned int dna =  shared[laneID].dna;
	if( randomFloat < mutationRate )
	{
		unsigned int gene_to_mutate=(unsigned int)(curand_uniform(&localState))*25;

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
	unsigned int count[32] = {};
	count[laneID] = 0;
	
	for( int i = 0; i < 25; i++ )
	{
		if( CheckBit(shared[laneID].dna,i) )
		{
			if( CheckBit(target, i))
			{
				count[laneID]++;
			}
		}
		else
		{
			if( !CheckBit(target, i))
			{
				count[laneID]++;
			}
		}
	}
	__syncwarp();
	if( count[laneID] == 0 )
	{ 
		shared[laneID].fitness = 0.05f;
	}
	else
	{
		if( count[laneID] >=24 ) {
		//printf("Finished in loop iteratons, block: %d, warp: %d, thread: %d", blockIdx.x, warpID, laneID);
		wd[warpID].complete = 1010101010;
		wd[warpID].dna = shared[laneID].dna;
		
		}
		else
		{
			shared[laneID].fitness = ((float)count[laneID]*count[laneID])/(25.0f);
		}
	}

	__syncwarp();

	// sort per warp
	int newID = warp_bitonic_sort_insect(laneID, shared);

	swap[laneID] = shared[newID];

	//shared[laneID] = swap[laneID];

	// finished with shared, so reduce it
	__syncwarp();

	warp_insect_all_reduce(shared, laneID);
	//warp_insect_scan(shared, laneID);

	__syncwarp();
	//if( laneID == 0 ) // power saving
	float warp_Reduced=	wd[blockIdx.x*(blockDim.x/WARP_SIZE)  + warpID].warp_reduced = shared[0].fitness;//in[warpID+WARP_SIZE-1].fitness;

	wd[blockIdx.x*(blockDim.x/WARP_SIZE) + warpID].warp_best = shared[WARP_SIZE-1];
	
	// breeding
	float breedingSelector = curand_uniform(&localState) * warp_Reduced*100;

	float cumulativeSum = 0.0f;
	int crossover_selection = 31;
	for( int j = 1; j < 32; j++ )
	{
		
		if(breedingSelector > cumulativeSum && breedingSelector < cumulativeSum + swap[j].fitness*100)
	  	{
			crossover_selection = j;
			
			break;
		}
		else cumulativeSum += swap[j].fitness*100;
	}

	__syncwarp();

	// cloning the best index
	shared[laneID].dna = (swap[laneID].dna & 0xffff0000) + (swap[crossover_selection].dna & 0x0000ffff);
	swap[laneID].dna = shared[laneID].dna;
	


	// thread migration
	//if( warpID-1 > -1 && laneID == 0 )
	//	out[warpID-1] = out[warpID];

	//__syncthreads();

	// WARP (laneID + 1) % 32; this reads the next lanes memory, but %32 is the remainder of division by 32 so when laneID=31 we get 32%32=0, and 33%32=1 etc, the last lane reads from the first

	// cloning

	//swap[0] =swap[WARP_SIZE-1];

	// store per warp data
	


	out[tid] = swap[laneID];
	
	in[tid] = out[tid];

//__syncwarp();

//	swap[laneID].dna = 0;
//	swap[laneID].fitness = 0.0f;

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


    insect_test(true);

  return 0;
}



unsigned int ANTENNAE_MASK = 0;
unsigned int HEAD_MASK = 0;
unsigned int WINGS_MASK = 0;
unsigned int BODY_MASK = 0;
unsigned int FEET_MASK = 0;
unsigned int BODY_COLOR_MASK = 0;
unsigned int SIZE_MASK = 0;
unsigned int HEAD_COLOR_MASK = 0;

unsigned int antennae_start = 0;
unsigned int head_start = 4;
unsigned int wing_start = 6;
unsigned int body_start = 10;
unsigned int feet_start = 16;
unsigned int body_color_start = 17;
unsigned int size_start = 20;
unsigned int head_color_start = 22;
unsigned int head_color_end = 25;

void ComputeMasks()
{
	for(unsigned int i = antennae_start; i < head_start; i++) SetBit(ANTENNAE_MASK,i);
	for(unsigned int i = head_start; i < wing_start; i++) SetBit(HEAD_MASK,i);
	for(unsigned int i = wing_start; i < body_start; i++) SetBit(WINGS_MASK,i);
	for(unsigned int i = body_start; i < feet_start; i++) SetBit(BODY_MASK,i);
	for(unsigned int i = feet_start; i < body_color_start; i++) SetBit(FEET_MASK,i);
	for(unsigned int i = body_color_start; i < size_start; i++) SetBit(BODY_COLOR_MASK,i);
	for(unsigned int i = size_start; i < head_color_start; i++) SetBit(SIZE_MASK,i);
	for(unsigned int i = head_color_start; i < head_color_end; i++) SetBit(HEAD_COLOR_MASK,i);
}

void SetAntennae(unsigned int *dna,unsigned  int choice)
{
	*dna |= (ANTENNAE_MASK &(choice) ) ;
}
void SetHead(unsigned int *dna, unsigned int choice)
{
	*dna |= (HEAD_MASK &(choice));
}
void SetWings(unsigned int *dna,unsigned  int choice)
{
	*dna |= (WINGS_MASK &(choice));
}
void SetBody(unsigned int *dna,unsigned  int choice)
{
	*dna |= (BODY_MASK &(choice));
}
void SetFeet(unsigned int *dna,unsigned  int choice)
{
	*dna |= (FEET_MASK &(choice));
}
void SetBodyColor(unsigned int *dna, unsigned int choice)
{
	*dna |= (BODY_COLOR_MASK &(choice));
}
void SetSize(unsigned int *dna, unsigned int choice)
{
	*dna |= (SIZE_MASK &(choice));
}
void SetHeadColor(unsigned int *dna,unsigned  int choice)
{
	*dna |= (HEAD_COLOR_MASK &(choice));
}

void SetInsect(unsigned int* dna)
{
 SetAntennae( dna, 2);
print_dna(*dna);
printf("\n");
 SetHead(dna, 4);
print_dna(*dna);printf("\n");
 SetWings( dna,34);
print_dna(*dna);printf("\n");
 SetBody( dna, 5);
print_dna(*dna);printf("\n");
 SetFeet( dna, 1);
print_dna(*dna);printf("\n");
 SetBodyColor( dna,3);
print_dna(*dna);printf("\n");
 SetSize( dna, 2);
print_dna(*dna);printf("\n");
SetHeadColor( dna, 1);	
print_dna(*dna);printf("\n");
	//return dna;
}

/**
 * Host main routine
 */
int
insect_test(bool b_verify)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;



    // Print the vector length to be used, and compute its size
    int numElements = 32768;
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
    	int threadsPerBlock = 256;
    	//int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	int blocksPerGrid =(numElements ) / threadsPerBlock;

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

	unsigned int insect_dna = 0;

	ComputeMasks();
	SetInsect(&insect_dna);


	/*ret_data *h_ret = (ret_data*)malloc(sizeof(ret_data));
	h_ret->complete = false;
	h_ret->warp_id = 0;
	h_ret->thread_id = 0;
	h_ret->dna = 0;

	ret_data *d_ret = 0;

	err = cudaMalloc((void **)&d_ret, sizeof(ret_data));

    	if (err != cudaSuccess)
    	{
        	printf("Failed to cudamalloc a ret_data ");
        	exit(EXIT_FAILURE); 
    	}

	err = cudaMemcpy(h_ret, d_ret,  sizeof(ret_data), cudaMemcpyHostToDevice);
    	if (err != cudaSuccess)
    	{
        	printf("Could not allocate memory on device for results: ");
        	exit(EXIT_FAILURE); 
    	}*/

	print_dna(insect_dna);
    

    	for( int p = 0; p < 50; p++ )
	{
		cudaEvent_t start,stop;
    		cudaEventCreate(&start);
    		cudaEventCreate(&stop);

    		cudaEventRecord(start);
    		printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


		

		fitness_samples<<<blocksPerGrid,threadsPerBlock>>>(d_A, insect_dna, d_C, d_rngStates, numElements, D_wd, numElements/32);

		cudaEventRecord(stop);
		
		printf("Copy output data from the CUDA device to the host memory\n");
    		err = cudaMemcpy(wdH, D_wd, sizeof(warp_data)*numElements / 32, cudaMemcpyDeviceToHost);
    		if (err != cudaSuccess)
    		{
        		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        		exit(EXIT_FAILURE);
    		}
    		



	    	cudaEventSynchronize(stop);

    		float milliseconds = 0;
    		cudaEventElapsedTime(&milliseconds, start, stop);

    		// size in bytes * 3 memory accesses, 2 retrieve, one store (unit converted)
    		printf("Effective Bandwidth (GB/s): %f\n", insect_size * 32 / milliseconds / 1e6 );  

    		// one floating point operation (add) * numElements (unit converted)
    		printf("Throughput (GFLOPS): %f\n", 5*numElements / milliseconds / 1e6 );  

    		printf( "Time Elapsed (ms): %f\n", milliseconds);

    		err = cudaGetLastError();

		bool target_reached = false;

		for( int i = 0; i < numElements / 32; i++ )
		{
			if( wdH[i].complete == 1010101010 )
			{
				target_reached = true;
				printf("TARGET_REACHED, warp_reduced: %f, warp_best: %f, \n", wdH[i].warp_reduced, wdH[i].warp_best.fitness);
				
				print_dna(wdH[i].warp_best.dna);
				printf(" \n");
				print_dna(insect_dna);
				break;
			}
			
		}

		if( target_reached )


		{
			printf("Target Reached at iteration %d", p);
			break;
		}
	}


    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, insect_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

   
    if( false )
    {
        // Verify that the result vector is correct
        for (int i = 1; i < numElements; ++i)
        {
          //  if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		/*if( h_C[i] < h_C[i-1] )
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                exit(EXIT_FAILURE);
            }*/
		//if(i%64==0)	print_dna( h_C[i].dna);printf("\n");
		//if(i%64==0)	
//printf("%f, ", (double)h_C[i].fitness);
	}

	for( int i = 0; i < numElements / 32; i++ )
	{
		printf("warp_reduced: %f, warp_best: %f, \n", wdH[i].warp_reduced, wdH[i].warp_best.fitness);
	}
    }

printf("\nh_C[0]: %f\n, ", h_C[0].fitness);

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

