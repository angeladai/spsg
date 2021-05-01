#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include "cuda_SimpleMatrixUtil.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024


//#define _DEBUG


namespace {
struct SdfData
{
	int dimz;
	int dimy;
	int dimx;
	int* sparse_mapping;
	long* locs;
	float* vals_sdf;
	float* vals_color;
	float* vals_normal;
};
struct RayCastData
{
	float* image_normal;
	float* image_depth;
	float* image_color;
	int* mapping3dto2d;
	int* mapping3dto2d_num;
	int maxPixelsPerVoxel;
};
struct RayCastParams
{
	int width;
	int height;
	float depthMin;
	float depthMax;
	float* intrinsicsParams;
	float threshSampleDist;
	float rayIncrement;

	__device__ inline
	float getFx(int batch) const { return intrinsicsParams[batch*4 + 0]; }
	__device__ inline
	float getFy(int batch) const { return intrinsicsParams[batch*4 + 1]; }
	__device__ inline
	float getMx(int batch) const { return intrinsicsParams[batch*4 + 2]; }
	__device__ inline
	float getMy(int batch) const { return intrinsicsParams[batch*4 + 3]; }
};
struct RayCastSample
{
	float sdf;
	float alpha;
	uint weight;
};

__device__
inline float3 kinectDepthToSkeleton(float mx, float my, float fx, float fy, uint ux, uint uy, float depth)	{
	const float x = ((float)ux-mx) / fx;
	const float y = ((float)uy-my) / fy;
	return make_float3(depth*x, depth*y, depth);
}
__device__
inline float kinectProjToCameraZ(float depthMin, float depthMax, float z) {
	return z * (depthMax - depthMin) + depthMin;
}
__device__
inline float3 kinectProjToCamera(float depthMin, float depthMax, float mx, float my, float fx, float fy, uint ux, uint uy, float z)	{
	float fSkeletonZ = kinectProjToCameraZ(depthMin, depthMax, z);
	return kinectDepthToSkeleton(mx, my, fx, fy, ux, uy, fSkeletonZ);
}

__device__
bool getVoxel(const SdfData sdf, int batch, const float3& pos, float& dist, float3& color, float3& normal) {
	int3 pi = make_int3(pos+make_float3(sign(pos))*0.5f);
	if (pi.x >= 0 && pi.y >= 0 && pi.z >= 0 && pi.x < sdf.dimx && pi.y < sdf.dimy && pi.z < sdf.dimz) {
		const int idx = sdf.sparse_mapping[batch*sdf.dimz*sdf.dimy*sdf.dimx + pi.z*sdf.dimy*sdf.dimx + pi.y*sdf.dimx + pi.x];
		if (idx == -1) return false;
		dist = sdf.vals_sdf[idx];
		color.x = sdf.vals_color[idx*3+0];
		color.y = sdf.vals_color[idx*3+1];
		color.z = sdf.vals_color[idx*3+2];
		normal.x = sdf.vals_normal[idx*3+0];
		normal.y = sdf.vals_normal[idx*3+1];
		normal.z = sdf.vals_normal[idx*3+2];
		return true;
	}
	return false;
}

__device__ 
bool trilinearInterpolationSimpleFastFast(const SdfData sdf, int batch, const float3& pos, float& dist, float3& color, float3& normal) {
	const float oSet = 1.0f; // voxel size -> 1
	const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
	float3 weight = fracf(pos); // worldtovoxel -> identity

	dist = 0.0f;
	float vsdf; float3 vcolor; float3 vnormal;
	color = make_float3(0.0f, 0.0f, 0.0f);
	normal = make_float3(0.0f, 0.0f, 0.0f);
	getVoxel(sdf, batch, pos, vsdf, color, normal);

	if (!getVoxel(sdf, batch, posDual+make_float3(0.0f, 0.0f, 0.0f), vsdf, vcolor, vnormal)) return false;
	dist += (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(oSet, 0.0f, 0.0f), vsdf, vcolor, vnormal)) return false;
	dist += weight.x *(1.0f-weight.y)*(1.0f-weight.z)*vsdf;
	
	if (!getVoxel(sdf, batch, posDual+make_float3(0.0f, oSet, 0.0f), vsdf, vcolor, vnormal)) return false;
	dist += (1.0f-weight.x)*weight.y *(1.0f-weight.z)*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(0.0f, 0.0f, oSet), vsdf, vcolor, vnormal)) return false;
	dist += (1.0f-weight.x)*(1.0f-weight.y)*weight.z*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(oSet, oSet, 0.0f), vsdf, vcolor, vnormal)) return false;
	dist += weight.x*weight.y *(1.0f-weight.z)*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(0.0f, oSet, oSet), vsdf, vcolor, vnormal)) return false;
	dist += (1.0f-weight.x)*weight.y *weight.z*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(oSet, 0.0f, oSet), vsdf, vcolor, vnormal)) return false;
	dist += weight.x*(1.0f-weight.y)*weight.z*vsdf;

	if (!getVoxel(sdf, batch, posDual+make_float3(oSet, oSet, oSet), vsdf, vcolor, vnormal)) return false;
	dist += weight.x*weight.y*weight.z*vsdf;

	return true;
}

__device__
float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar)
{
	return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
}

// d0 near, d1 far
__device__
	bool findIntersectionBisection(const SdfData sdf, int batch, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, float3& color, float3& normal)
{
	float a = r0; float aDist = d0;
	float b = r1; float bDist = d1;
	float c = 0.0f;

#pragma unroll 1
	for(uint i = 0; i<3; i++) 
	{
		c = findIntersectionLinear(a, b, aDist, bDist);

		float cDist;
		if(!trilinearInterpolationSimpleFastFast(sdf, batch, worldCamPos+c*worldDir, cDist, color, normal)) return false;

		if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
		else { b = c; bDist = cDist; }
	}

	alpha = c;

	return true;
}

__device__
void traverseCoarseGridSimpleSampleAll(const SdfData sdf, int batch, RayCastData raycastData, 
	const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, const RayCastParams& params)
{
	RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0;
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length

	float rayCurrent = depthToRayLength * params.depthMin;	// Convert depth to raylength
	float rayEnd = depthToRayLength * params.depthMax;		// Convert depth to raylength

#pragma unroll 1
	while(rayCurrent < rayEnd) {
		float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
		float dist;	float3 color; float3 normal;
		if(trilinearInterpolationSimpleFastFast(sdf, batch, currentPosWorld, dist, color, normal))
		{
			if(lastSample.weight > 0 && ((lastSample.sdf > 0.0f && dist < 0.0f) || (lastSample.sdf < 0.0f && dist > 0.0f))) // current sample is always valid here 
			{
				float alpha; float3 color2; float3 normal2;
				bool b = findIntersectionBisection(sdf, batch, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2, normal2);
				
				float3 currentIso = worldCamPos+alpha*worldDir;
				if(b && abs(lastSample.sdf - dist) < params.threshSampleDist)
				{
					if(abs(dist) < params.threshSampleDist)
					{
						float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

						raycastData.image_color[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 0] = color2.x;
						raycastData.image_color[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 1] = color2.y;
						raycastData.image_color[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 2] = color2.z;
						if (!(normal2.x == 0 && normal2.y == 0 && normal2.z == 0)) {
							raycastData.image_normal[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 0] = normal2.x;
							raycastData.image_normal[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 1] = normal2.y;
							raycastData.image_normal[(dTid.z*params.width*params.height + dTid.y*params.width+dTid.x)*3 + 2] = normal2.z;
						}
						raycastData.image_depth[dTid.z*params.width*params.height + dTid.y*params.width+dTid.x] = depth;

						const int3 loc = make_int3(currentIso+make_float3(sign(currentIso))*0.5f);
						const int idx = sdf.sparse_mapping[dTid.z*sdf.dimz*sdf.dimy*sdf.dimx + loc.z*sdf.dimy*sdf.dimx + loc.y*sdf.dimx + loc.x];
						if (idx < 0) printf("ERROR currentIso (%f,%f,%f)->(%d,%d,%d) ==> %d from sparse_mapping\n", currentIso.x, currentIso.y, currentIso.z, loc.x, loc.y, loc.z, idx);						
						int offset = atomicAdd(&raycastData.mapping3dto2d_num[idx], 1);
						if (offset < raycastData.maxPixelsPerVoxel)	{
							raycastData.mapping3dto2d[idx*raycastData.maxPixelsPerVoxel+offset] = dTid.y*params.width+dTid.x;
						}
						
						return;
					}
				}
			}
	
			lastSample.sdf = dist;
			lastSample.alpha = rayCurrent;
			lastSample.weight = 1;
			rayCurrent += params.rayIncrement;
		} else {
			lastSample.weight = 0;
			rayCurrent += params.rayIncrement;
		}
	}
}

__global__ void raycast_rgbd_cuda_kernel(
    const SdfData sdf,
    const float* __restrict__ viewMatrixInv,
    RayCastData raycastData,
    const RayCastParams params) {
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if (x < params.width && y < params.height /*&& batch < batch_size*/) {
		// init
		raycastData.image_color[(batch*params.width*params.height + y*params.width + x)*3 + 0] = MINF;
		raycastData.image_color[(batch*params.width*params.height + y*params.width + x)*3 + 1] = MINF;
		raycastData.image_color[(batch*params.width*params.height + y*params.width + x)*3 + 2] = MINF;
		raycastData.image_depth[batch*params.width*params.height + y*params.width + x] = MINF;
		raycastData.image_normal[(batch*params.width*params.height + y*params.width + x)*3 + 0] = MINF;
		raycastData.image_normal[(batch*params.width*params.height + y*params.width + x)*3 + 1] = MINF;
		raycastData.image_normal[(batch*params.width*params.height + y*params.width + x)*3 + 2] = MINF;

		const float4x4 curViewMatrixInv = *(float4x4*)(viewMatrixInv + batch*16);
		float3 camDir = normalize(kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch), params.getFx(batch), params.getFy(batch), x, y, 1.0f));
		float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);
		float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
		float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

		traverseCoarseGridSimpleSampleAll(sdf, batch, raycastData, worldCamPos, worldDir, camDir, make_int3(x,y,batch), params);
	}
}


__device__
void traverseOccGrid(const uint8_t* occ3d, uint8_t* occ2d, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, const RayCastParams& params, int dimz, int dimy, int dimx)
{
	const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length
	float rayCurrent = depthToRayLength * params.depthMin;	// Convert depth to raylength
	float rayEnd = depthToRayLength * params.depthMax;		// Convert depth to raylength
#pragma unroll 1
	while(rayCurrent < rayEnd) {
		float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
		int3 pos = make_int3(currentPosWorld+make_float3(sign(currentPosWorld))*0.5f);
		if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < dimx && pos.y < dimy && pos.z < dimz) {
			if (occ3d[dTid.z*dimz*dimy*dimx + pos.z*dimy*dimx + pos.y*dimx + pos.x] != 0) {
				occ2d[dTid.z*params.width*params.height + dTid.y*params.width+dTid.x] = 1;
				return;
			}
		}
		rayCurrent += params.rayIncrement;
	}
}

__global__ void raycast_occ_cuda_kernel(
    const uint8_t* __restrict__ occ3d,
    uint8_t* __restrict__ occ2d,
    const float* __restrict__ viewMatrixInv,
    const RayCastParams params,
	int dimz, 
	int dimy, 
	int dimx) {
	const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int batch = blockIdx.z;

	if (x < params.width && y < params.height /*&& batch < params.batch_size*/) {
		// init
		occ2d[batch*params.width*params.height + y*params.width + x] = 0;

		const float4x4 curViewMatrixInv = *(float4x4*)(viewMatrixInv + batch*16);
		float3 camDir = normalize(kinectProjToCamera(params.depthMin, params.depthMax, params.getMx(batch), params.getMy(batch), params.getFx(batch), params.getFy(batch), x, y, 1.0f));
		float3 worldCamPos = curViewMatrixInv * make_float3(0.0f, 0.0f, 0.0f);
		float4 w = curViewMatrixInv * make_float4(camDir, 0.0f);
		float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

		traverseOccGrid(occ3d, occ2d, worldCamPos, worldDir, camDir, make_int3(x,y,batch), params, dimz, dimy, dimx);
	}
}

__global__ void construct_dense_sparse_mapping_kernel(
	const long* __restrict__ locs,
	int* __restrict__ sparse_mapping,
	int dimz,
	int dimy, 
	int dimx,
	int N) {
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N /*&& batch < batch_size*/) {
		const long z = locs[idx * 4 + 0];
		const long y = locs[idx * 4 + 1];
		const long x = locs[idx * 4 + 2];
		const long batch = locs[idx * 4 + 3];
		sparse_mapping[batch * dimz*dimy*dimx + z * dimy*dimx + y * dimx + x] = idx;
	}
}


__global__ void raycast_rgbd_cuda_backward_kernel(
	const float* __restrict__ grad_color,
	const float* __restrict__ grad_depth,
	const float* __restrict__ grad_normal,
	const int* __restrict__ sparse_mapping,
	const int* __restrict__ mapping3dto2d,
	const int* __restrict__ mapping3dto2d_num,
	float* __restrict__ d_color,
	float* __restrict__ d_depth,
	float* __restrict__ d_normals,
	int image_height,
	int image_width,
	int dimz,
	int dimy, 
	int dimx,
	int maxPixelsPerVoxel) {
	const int z = blockIdx.x;
	const int yx = blockIdx.y;
	const int y = yx / dimx;
	const int x = yx % dimx;
	const int batch = blockIdx.z;
	const int tidx = threadIdx.x;

	if (x >= 0 && y >= 0 && z >= 0 && x < dimx && y < dimy && z < dimz && tidx < maxPixelsPerVoxel /*&& batch < batch_size*/) {
		const int idx = sparse_mapping[batch * dimz*dimy*dimx + z * dimy*dimx + y * dimx + x];
		if (idx >= 0) {
			const int num = mapping3dto2d_num[idx];
			if (tidx < num) {
				const int pidx = mapping3dto2d[idx*maxPixelsPerVoxel + tidx];
				const int py = pidx / image_width;
				const int px = pidx % image_width;
				const float val0 = grad_color[(batch*(image_height*image_width) + py*image_width + px)*3 + 0] / (float)min(num,maxPixelsPerVoxel);
				const float val1 = grad_color[(batch*(image_height*image_width) + py*image_width + px)*3 + 1] / (float)min(num,maxPixelsPerVoxel);
				const float val2 = grad_color[(batch*(image_height*image_width) + py*image_width + px)*3 + 2] / (float)min(num,maxPixelsPerVoxel);
				const float vn0 = grad_normal[(batch*(image_height*image_width) + py*image_width + px)*3 + 0] / (float)min(num,maxPixelsPerVoxel);
				const float vn1 = grad_normal[(batch*(image_height*image_width) + py*image_width + px)*3 + 1] / (float)min(num,maxPixelsPerVoxel);
				const float vn2 = grad_normal[(batch*(image_height*image_width) + py*image_width + px)*3 + 2] / (float)min(num,maxPixelsPerVoxel);
				const float vald = grad_depth[batch*(image_height*image_width) + py*image_width + px] / (float)min(num,maxPixelsPerVoxel);

				atomicAdd(&d_color[idx*3 + 0], val0);
				atomicAdd(&d_color[idx*3 + 1], val1);
				atomicAdd(&d_color[idx*3 + 2], val2);
				
				atomicAdd(&d_normals[idx*3 + 0], vn0);
				atomicAdd(&d_normals[idx*3 + 1], vn1);
				atomicAdd(&d_normals[idx*3 + 2], vn2);
				
				atomicAdd(&d_depth[idx], vald);
			} // has mapping
		} // valid loc
	} // in 3d dims
}
} // namespace

void raycast_rgbd_cuda_forward(
    torch::Tensor sparse_mapping,
    torch::Tensor locs,
	torch::Tensor vals_sdf,
	torch::Tensor vals_color,
	torch::Tensor vals_normal,
    torch::Tensor viewMatrixInv, // batched
    torch::Tensor imageColor,
    torch::Tensor imageDepth,
    torch::Tensor imageNormal,
	torch::Tensor mapping3dto2d,
	torch::Tensor mapping3dto2d_num,
    torch::Tensor intrinsicParams, // batched bx4 (fx,fy,mx,my)
    torch::Tensor opts) { //depthmin,depthmax,threshSampleDist,rayIncrement

	auto opts_accessor = opts.accessor<float,1>();
	RayCastParams params;
	params.width = (int)(opts_accessor[0]+0.5f);
	params.height = (int)(opts_accessor[1]+0.5f);
	params.depthMin = opts_accessor[2];
	params.depthMax = opts_accessor[3];
	params.threshSampleDist = opts_accessor[4];
	params.rayIncrement = opts_accessor[5];
	params.intrinsicsParams = intrinsicParams.data<float>();

	SdfData sdf;
	sdf.dimx = (int)(opts_accessor[6]+0.5f);
	sdf.dimy = (int)(opts_accessor[7]+0.5f);
	sdf.dimz = (int)(opts_accessor[8]+0.5f);
	sdf.sparse_mapping = sparse_mapping.data<int>();
	sdf.locs = locs.data<long>();
	sdf.vals_sdf = vals_sdf.data<float>();
	sdf.vals_color = vals_color.data<float>();
	sdf.vals_normal = vals_normal.data<float>();

	RayCastData raycastData;
	raycastData.image_normal = imageNormal.data<float>();
	raycastData.image_depth = imageDepth.data<float>();
	raycastData.image_color = imageColor.data<float>();
	raycastData.mapping3dto2d = mapping3dto2d.data<int>();
	raycastData.mapping3dto2d_num = mapping3dto2d_num.data<int>();
	raycastData.maxPixelsPerVoxel = (int)mapping3dto2d.size(1);

	{
		const int num = mapping3dto2d.size(0) * mapping3dto2d.size(1);
		cutilSafeCall(cudaMemset(raycastData.mapping3dto2d, -1, sizeof(int)*num));
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
	{
		const int num = mapping3dto2d_num.size(0);
		cutilSafeCall(cudaMemset(raycastData.mapping3dto2d_num, 0, sizeof(int)*num));
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}

	const int batch_size = sparse_mapping.size(0);
	const dim3 gridSize((params.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (params.height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

    raycast_rgbd_cuda_kernel<<<gridSize, blockSize>>>(
        sdf,
        viewMatrixInv.data<float>(),
        raycastData,
        params);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


void construct_dense_sparse_mapping_cuda(
    torch::Tensor locs,
    torch::Tensor sparse_mapping) {
	int num = (int)locs.size(0);
	int batch_size = sparse_mapping.size(0);
	int dimz = sparse_mapping.size(1);
	int dimy = sparse_mapping.size(2);
	int dimx = sparse_mapping.size(3);

	cutilSafeCall(cudaMemset(sparse_mapping.data<int>(), -1, sizeof(int)*batch_size*dimz*dimy*dimx));
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	{
		const dim3 gridSize((num + 64 - 1)/64);
		const dim3 blockSize(64);
		construct_dense_sparse_mapping_kernel<<<gridSize, blockSize>>>(
			locs.data<long>(),
			sparse_mapping.data<int>(),
			dimz, dimy, dimx,
			num);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
}

void raycast_rgbd_cuda_backward(
    torch::Tensor grad_color,
    torch::Tensor grad_depth,
	torch::Tensor grad_normal,
    torch::Tensor sparse_mapping,
    torch::Tensor mapping3dto2d,
	torch::Tensor mapping3dto2d_num,
	torch::Tensor dims,
	torch::Tensor d_color,
	torch::Tensor d_depth,
	torch::Tensor d_normals) {
	auto dims_accessor = dims.accessor<int,1>();
	const int image_height = grad_color.size(1);
	const int image_width = grad_color.size(2);
	const int batch_size = dims_accessor[0];
	const int dimx = dims_accessor[1];
	const int dimy = dims_accessor[2];
	const int dimz = dims_accessor[3];
	const int maxPixelsPerVoxel = (int)mapping3dto2d.size(1);

	cutilSafeCall(cudaMemset(d_color.data<float>(), 0, sizeof(float)*d_color.size(0)*d_color.size(1)));
	cutilSafeCall(cudaMemset(d_depth.data<float>(), 0, sizeof(float)*d_depth.size(0)*d_depth.size(1)));
	cutilSafeCall(cudaMemset(d_normals.data<float>(), 0, sizeof(float)*d_normals.size(0)*d_normals.size(1)));
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif

	const dim3 gridSize(dimz, dimy*dimx, batch_size);
	const dim3 blockSize(64);
	raycast_rgbd_cuda_backward_kernel<<<gridSize, blockSize>>>(
			grad_color.data<float>(),
			grad_depth.data<float>(),
			grad_normal.data<float>(),
			sparse_mapping.data<int>(),
			mapping3dto2d.data<int>(),
			mapping3dto2d_num.data<int>(),
			d_color.data<float>(),
			d_depth.data<float>(),
			d_normals.data<float>(),
			image_height, image_width,
			dimz, dimy, dimx, maxPixelsPerVoxel);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


void raycast_occ_cuda_forward(
    at::Tensor occ3d,
    at::Tensor occ2d,
    at::Tensor viewMatrixInv, // batched
    at::Tensor intrinsicParams, // batched bx4 (fx,fy,mx,my)
    at::Tensor opts) { //depthmin,depthmax,rayIncrement

	auto opts_accessor = opts.accessor<float,1>();
	RayCastParams params;
	params.width = (int)(opts_accessor[0]+0.5f);
	params.height = (int)(opts_accessor[1]+0.5f);
	params.depthMin = opts_accessor[2];
	params.depthMax = opts_accessor[3];
	params.rayIncrement = opts_accessor[4];
	params.intrinsicsParams = intrinsicParams.data<float>();

	const int batch_size = occ3d.size(0);
	const int dimz = occ3d.size(2);
	const int dimy = occ3d.size(3);
	const int dimx = occ3d.size(4);
	const dim3 gridSize((params.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (params.height + T_PER_BLOCK - 1)/T_PER_BLOCK, batch_size);
	const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    raycast_occ_cuda_kernel<<<gridSize, blockSize>>>(
        occ3d.data<uint8_t>(),
        occ2d.data<uint8_t>(),
        viewMatrixInv.data<float>(),
        params,
		dimz, 
		dimy, 
		dimx);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
