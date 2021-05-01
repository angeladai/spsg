#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include "cuda_SimpleMatrixUtil.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024


namespace {

__device__ __forceinline__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

__device__ __forceinline__ float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f-(dist*dist)/(2.0*sigma*sigma)));
}

__device__ __forceinline__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

__device__ __forceinline__ float gaussD(float sigma, int x)
{
	return exp(-((x*x)/(2.0f*sigma*sigma)));
}
__device__ __forceinline__ float3 kinectDepthToSkeleton(uint ux, uint uy, float depth, float fx, float fy, float mx, float my)	{
	const float x = ((float)ux-mx) / fx;
	const float y = ((float)uy-my) / fy;
	return make_float3(depth*x, depth*y, depth);
}

__global__ void bilateral_filter_floatmap_kernel(
	torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> output,
	const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> input,
	float sigmaD,
	float sigmaR,
	unsigned int batch_size,
	unsigned int width,
	unsigned int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if(x >= width || y >= height || batch >= batch_size) return;

	const int kernelRadius = (int)ceil(2.0*sigmaD);

	output[batch][0][y][x] = 0.0f;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const auto depthCenter = input[batch][0][y][x];
	if(depthCenter != MINF && depthCenter != 0)
	{
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{		
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = input[batch][0][n][m];

					if (currentDepth != MINF && currentDepth != 0) {
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, currentDepth-depthCenter);

						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}

		if(sumWeight > 0.0f) output[batch][0][y][x] = sum / sumWeight;
	}
}

#define STRUCTURE_SIZE 5
__global__ void median_fill_depthmap_kernel(
	torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> output,
	const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> input,
	int batch_size,
	int width,
	int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if(x >= 0 && x < width && y >= 0 && y < height && batch >= 0 && batch < batch_size) {
		const float cur = input[batch][0][y][x];
		if (cur != MINF && cur != 0) {
			output[batch][0][y][x] = cur;
		}
		else {
			const int diameter = 2*STRUCTURE_SIZE+1;
			int discreteDepthVals[diameter*diameter];
	
			// collect values
			int numValid = 0;
			for(int i = -STRUCTURE_SIZE; i<=STRUCTURE_SIZE; i++) {
				for(int j = -STRUCTURE_SIZE; j<=STRUCTURE_SIZE; j++) {
					if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height) {
						float depth = input[batch][0][y+i][x+j];
						if(depth != MINF && depth != 0.0f) {
							discreteDepthVals[(i+STRUCTURE_SIZE)*diameter+(j+STRUCTURE_SIZE)] = (int)(1000*depth+0.5f);
							numValid++;
						}
						else discreteDepthVals[(i+STRUCTURE_SIZE)*diameter+(j+STRUCTURE_SIZE)] = -1;
					}
					else discreteDepthVals[(i+STRUCTURE_SIZE)*diameter+(j+STRUCTURE_SIZE)] = -1;
				}
			}
			
			// sort
			for (int i = 0; i < diameter*diameter; i++) {
				for (int j = i + 1; j < diameter*diameter; j++) {
					if (discreteDepthVals[i] > discreteDepthVals[j]) {
						int tmp = discreteDepthVals[i];
						discreteDepthVals[i] = discreteDepthVals[j];
						discreteDepthVals[j] = tmp;
					}
				}
			}
			int val = discreteDepthVals[diameter*diameter-numValid + (numValid+1)/2];
			if (val <= 0) output[batch][0][y][x] = 0;
			else output[batch][0][y][x] = 0.001f * (float)val;
		}
	}
}

__global__ void convert_depth_to_cameraspace_kernel(
	torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> output,
	const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> input,
	const torch::PackedTensorAccessor<float,2,torch::RestrictPtrTraits,size_t> intrinsics,
	unsigned int batch_size,
	unsigned int width,
	unsigned int height,
	float depthMin,
	float depthMax)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if (x < width && y < height && batch < batch_size) {
		output[batch][y][x][0] = 0.0f;
		output[batch][y][x][1] = 0.0f;
		output[batch][y][x][2] = 0.0f;

		float depth = input[batch][0][y][x];

		if(depth != 0) {
			float3 p = kinectDepthToSkeleton(x, y, depth, intrinsics[batch][0], intrinsics[batch][1], intrinsics[batch][2], intrinsics[batch][3]);
			output[batch][y][x][0] = p.x;
			output[batch][y][x][1] = p.y;
			output[batch][y][x][2] = p.z;
		}
	}
}

__global__ void compute_normals_kernel(
	torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> output,
	const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> input,
	int batch_size,
	int width,
	int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if(x >= width || y >= height || batch >= batch_size) return;

	output[batch][y][x][0] = 0.0f;
	output[batch][y][x][1] = 0.0f;
	output[batch][y][x][2] = 0.0f;

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float3 CC = make_float3(input[batch][y+0][x+0][0], input[batch][y+0][x+0][1], input[batch][y+0][x+0][2]);
		const float3 PC = make_float3(input[batch][y+1][x+0][0], input[batch][y+1][x+0][1], input[batch][y+1][x+0][2]);
		const float3 CP = make_float3(input[batch][y+0][x+1][0], input[batch][y+0][x+1][1], input[batch][y+0][x+1][2]);
		const float3 MC = make_float3(input[batch][y-1][x+0][0], input[batch][y-1][x+0][1], input[batch][y-1][x+0][2]);
		const float3 CM = make_float3(input[batch][y+0][x-1][0], input[batch][y+0][x-1][1], input[batch][y+0][x-1][2]);

		if( (CC.x != 0 || PC.x != 0 || CP.x != 0 || MC.x != 0 || CM.x != 0) && (CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF) )
		{
			const float3 n = cross(PC-MC, CP-CM);
			const float  l = length(n);

			if(l > 0.0f)
			{
				float3 out = n/-l;
				output[batch][y][x][0] = out.x;
				output[batch][y][x][1] = out.y;
				output[batch][y][x][2] = out.z;
			}
		}
	}
}

} // namespace

void bilateral_filter_floatmap_cuda(
    torch::Tensor output,
    torch::Tensor input,
	float sigmaD,
	float sigmaR) {

	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES(input.type(), "bilateral_filter_floatmap_kernel", ([&] {
		bilateral_filter_floatmap_kernel<<<gridSize, blockSize>>>(
			output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			sigmaD,
			sigmaR,
			batch_size,
			width,
			height);
  }));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

void median_fill_depthmap_cuda(
    torch::Tensor output,
    torch::Tensor input) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES(input.type(), "median_fill_depthmap_kernel", ([&] {
		median_fill_depthmap_kernel<<<gridSize, blockSize>>>(
			output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			batch_size,
			width,
			height);
  }));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	//fflush(stdout);
}

void compute_normals_cuda(
    torch::Tensor output,
    torch::Tensor input) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(1);
	const unsigned int width = (unsigned int)input.size(2);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES(input.type(), "compute_normals_kernel", ([&] {
		compute_normals_kernel<<<gridSize, blockSize>>>(
			output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			batch_size,
			width,
			height);
  }));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

void convert_depth_to_cameraspace_cuda(
    torch::Tensor output,
    torch::Tensor input,
	torch::Tensor intrinsics,
	float depthMin,
	float depthMax) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

	AT_DISPATCH_FLOATING_TYPES(input.type(), "convert_depth_to_cameraspace_kernel", ([&] {
		convert_depth_to_cameraspace_kernel<<<gridSize, blockSize>>>(
			output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
			intrinsics.packed_accessor<float,2,torch::RestrictPtrTraits,size_t>(),
			batch_size,
			width,
			height,
			depthMin,
			depthMax);
  }));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
