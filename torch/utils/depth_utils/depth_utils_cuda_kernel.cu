#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include "cuda_SimpleMatrixUtil.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024


namespace {

inline __device__ float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

inline __device__ float linearR(float sigma, float dist)
{
	return max(1.0f, min(0.0f, 1.0f-(dist*dist)/(2.0*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

inline __device__ float gaussD(float sigma, int x)
{
	return exp(-((x*x)/(2.0f*sigma*sigma)));
}
inline __device__ float3 kinectDepthToSkeleton(uint ux, uint uy, float depth, float fx, float fy, float mx, float my)	{
	const float x = ((float)ux-mx) / fx;
	const float y = ((float)uy-my) / fy;
	return make_float3(depth*x, depth*y, depth);
}

__global__ void bilateral_filter_floatmap_kernel(
	float* __restrict__ output,
	const float* __restrict__ input,
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

	//output[batch*height*width + y*width + x] = MINF;
	output[batch*height*width + y*width + x] = 0;

	float sum = 0.0f;
	float sumWeight = 0.0f;

	const float depthCenter = input[batch*height*width + y*width + x];
	if(depthCenter != MINF && depthCenter != 0)
	{
		for(int m = x-kernelRadius; m <= x+kernelRadius; m++)
		{
			for(int n = y-kernelRadius; n <= y+kernelRadius; n++)
			{		
				if(m >= 0 && n >= 0 && m < width && n < height)
				{
					const float currentDepth = input[batch*height*width + n*width+m];

					if (currentDepth != MINF && currentDepth != 0) {
						const float weight = gaussD(sigmaD, m-x, n-y)*gaussR(sigmaR, currentDepth-depthCenter);

						sumWeight += weight;
						sum += weight*currentDepth;
					}
				}
			}
		}

		if(sumWeight > 0.0f) output[batch*height*width + y*width + x] = sum / sumWeight;
	}
}

#define STRUCTURE_SIZE 5
__global__ void median_fill_depthmap_kernel(
	float* __restrict__ output,
	const float* __restrict__ input,
	int batch_size,
	int width,
	int height)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if(x >= 0 && x < width && y >= 0 && y < height && batch >= 0 && batch < batch_size) {
		const float cur = input[batch*height*width+y*width+x];
		if (cur != MINF && cur != 0) {
			output[batch*height*width+y*width+x] = cur;
		}
		else {
			const int diameter = 2*STRUCTURE_SIZE+1;
			int discreteDepthVals[diameter*diameter];
	
			// collect values
			int numValid = 0;
			for(int i = -STRUCTURE_SIZE; i<=STRUCTURE_SIZE; i++) {
				for(int j = -STRUCTURE_SIZE; j<=STRUCTURE_SIZE; j++) {
					if(x+j >= 0 && x+j < width && y+i >= 0 && y+i < height) {
						float depth = input[batch*height*width+(y+i)*width+(x+j)];
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
			//if (y == 238 && x == 167) {
			//	printf("(%d,%d) -> %d (numvalid %d)\n", y, x, val, numValid);
			//	for (int i = 0; i < diameter; i++) {
			//		printf("\t");
			//		for (int j = 0; j < diameter; j++)
			//			printf("%d, ", discreteDepthVals[i*diameter+j]);
			//		printf("\n");
			//	}
			//}
			if (val <= 0) output[batch*height*width+y*width+x] = 0;//output[batch*height*width+y*width+x] = MINF;
			else output[batch*height*width+y*width+x] = 0.001f * (float)val;
		}
	}
}

__global__ void convert_depth_to_cameraspace_kernel(
	float* __restrict__ output,
	const float* __restrict__ input,
	const float* __restrict__ intrinsics,
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
		float3* output3 = (float3*)output;
		output3[batch*height*width+y*width+x] = make_float3(0,0,0);

		float depth = input[batch*height*width+y*width+x];

		if(depth != 0) {
			output3[batch*height*width+y*width+x] = kinectDepthToSkeleton(x, y, depth, intrinsics[batch*4+0], intrinsics[batch*4+1], intrinsics[batch*4+2], intrinsics[batch*4+3]);
		}
	}
}

__global__ void compute_normals_kernel(
	float* __restrict__ output,
	const float* __restrict__ input,
	int batch_size,
	int width,
	int height)
{
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if(x >= width || y >= height || batch >= batch_size) return;

	float3* output3 = (float3*)output;
	output3[batch*height*width+y*width+x] = make_float3(0,0,0);

	if(x > 0 && x < width-1 && y > 0 && y < height-1)
	{
		const float3* input3 = (float3*)input;
		const float3 CC = input3[batch*height*width+(y+0)*width+(x+0)];
		const float3 PC = input3[batch*height*width+(y+1)*width+(x+0)];
		const float3 CP = input3[batch*height*width+(y+0)*width+(x+1)];
		const float3 MC = input3[batch*height*width+(y-1)*width+(x+0)];
		const float3 CM = input3[batch*height*width+(y+0)*width+(x-1)];

		if( (CC.x != 0 || PC.x != 0 || CP.x != 0 || MC.x != 0 || CM.x != 0) && (CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF) )
		{
			const float3 n = cross(PC-MC, CP-CM);
			const float  l = length(n);

			if(l > 0.0f)
			{
				output3[batch*height*width+y*width+x] = n/-l;

				//if (batch == 1 && y == 120 && x == 200) printf("normals[%d,%d,%d] = (%f,%f,%f) | (%f,%f,%f)^(%f,%f,%f)\n", batch,y,x, n.x/-l, n.y/-l, n.z/-l, PC.x-MC.x, PC.y-MC.y, PC.z-MC.z, CP.x-CM.x, CP.y-CM.y, CP.z-CM.z);
			}
		}
	}
}

} // namespace

void bilateral_filter_floatmap_cuda(
    at::Tensor output,
    at::Tensor input,
	float sigmaD,
	float sigmaR) {

	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);
	//printf("batch_size %d, height %d, width %d\n", batch_size, height, width);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

    bilateral_filter_floatmap_kernel<<<gridSize, blockSize>>>(
        output.data<float>(),
        input.data<float>(),
        sigmaD,
        sigmaR,
		batch_size,
		width,
		height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

void median_fill_depthmap_cuda(
    at::Tensor output,
    at::Tensor input) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);
	//printf("batch_size %d, height %d, width %d\n", batch_size, height, width);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

    median_fill_depthmap_kernel<<<gridSize, blockSize>>>(
        output.data<float>(),
        input.data<float>(),
		batch_size,
		width,
		height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	//fflush(stdout);
}

void compute_normals_cuda(
    at::Tensor output,
    at::Tensor input) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(1);
	const unsigned int width = (unsigned int)input.size(2);
	//printf("batch_size %d, height %d, width %d\n", batch_size, height, width);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

    compute_normals_kernel<<<gridSize, blockSize>>>(
        output.data<float>(),
        input.data<float>(),
		batch_size,
		width,
		height);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

void convert_depth_to_cameraspace_cuda(
    at::Tensor output,
    at::Tensor input,
	at::Tensor intrinsics,
	float depthMin,
	float depthMax) {
	const unsigned int batch_size = (unsigned int)input.size(0);
	const unsigned int height = (unsigned int)input.size(2);
	const unsigned int width = (unsigned int)input.size(3);
	//printf("batch_size %d, height %d, width %d\n", batch_size, height, width);

	const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

    convert_depth_to_cameraspace_kernel<<<gridSize, blockSize>>>(
        output.data<float>(),
        input.data<float>(),
		intrinsics.data<float>(),
		batch_size,
		width,
		height,
		depthMin,
		depthMax);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


