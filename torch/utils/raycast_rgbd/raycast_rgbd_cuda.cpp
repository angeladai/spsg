#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void raycast_rgbd_cuda_forward( 
    torch::Tensor sparse_mapping,
    torch::Tensor locs,
	torch::Tensor vals_sdf,
	torch::Tensor vals_color,
	torch::Tensor vals_normals,
    torch::Tensor viewMatrixInv,
    torch::Tensor imageColor,
    torch::Tensor imageDepth,
    torch::Tensor imageNormal,
	torch::Tensor mapping2dto3d,
	torch::Tensor mapping3dto2d_num,
	torch::Tensor intrinsicParams,
    torch::Tensor opts);

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
	torch::Tensor d_normals);
	
void raycast_occ_cuda_forward(
    at::Tensor occ3d,
    at::Tensor occ2d,
    at::Tensor viewMatrixInv, 
    at::Tensor intrinsicParams,
    at::Tensor opts);

void construct_dense_sparse_mapping_cuda(
    torch::Tensor locs,
    torch::Tensor sparse_mapping);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void raycast_color_forward(
    torch::Tensor sparse_mapping,
    torch::Tensor locs,
	torch::Tensor vals_sdf,
	torch::Tensor vals_color,
	torch::Tensor vals_normals,
    torch::Tensor viewMatrixInv,
    torch::Tensor imageColor,
    torch::Tensor imageDepth,
    torch::Tensor imageNormal,
	torch::Tensor mapping3dto2d,
	torch::Tensor mapping3dto2d_num,
	torch::Tensor intrinsicParams,
    torch::Tensor opts) {
  CHECK_INPUT(sparse_mapping);
  CHECK_INPUT(locs);
  CHECK_INPUT(vals_sdf);
  CHECK_INPUT(vals_color);
  CHECK_INPUT(vals_normals);
  CHECK_INPUT(viewMatrixInv);
  CHECK_INPUT(imageColor);
  CHECK_INPUT(imageDepth);
  CHECK_INPUT(imageNormal);
  CHECK_INPUT(mapping3dto2d);
  CHECK_INPUT(mapping3dto2d_num);
  CHECK_INPUT(intrinsicParams);

  return raycast_rgbd_cuda_forward(sparse_mapping, locs, vals_sdf, vals_color, vals_normals, viewMatrixInv, imageColor, imageDepth, imageNormal, mapping3dto2d, mapping3dto2d_num, intrinsicParams, opts);
}

void construct_dense_sparse_mapping(
    torch::Tensor locs,
    torch::Tensor sparse_mapping) {
  CHECK_INPUT(locs);
  CHECK_INPUT(sparse_mapping);

  return construct_dense_sparse_mapping_cuda(locs, sparse_mapping);
}

void raycast_color_backward(
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
  CHECK_INPUT(grad_color);
  CHECK_INPUT(grad_depth);
  CHECK_INPUT(grad_normal);
  CHECK_INPUT(sparse_mapping);
  CHECK_INPUT(mapping3dto2d);
  CHECK_INPUT(mapping3dto2d_num);
  CHECK_INPUT(d_color);
  CHECK_INPUT(d_depth);
  CHECK_INPUT(d_normals);

  return raycast_rgbd_cuda_backward(
      grad_color,
	  grad_depth,
	  grad_normal,
	  sparse_mapping,
      mapping3dto2d,
	  mapping3dto2d_num,
	  dims,
	  d_color,
	  d_depth,
	  d_normals);
}

void raycast_occ_forward(
    at::Tensor occ3d,
    at::Tensor occ2d,
    at::Tensor viewMatrixInv, 
    at::Tensor intrinsicParams,
    at::Tensor opts) {
  CHECK_INPUT(occ3d);
  CHECK_INPUT(occ2d);
  CHECK_INPUT(viewMatrixInv);
  CHECK_INPUT(intrinsicParams);
  return raycast_occ_cuda_forward(occ3d, occ2d, viewMatrixInv, intrinsicParams, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &raycast_color_forward, "raycast_color forward (CUDA)");
  m.def("backward", &raycast_color_backward, "raycast_color backward (CUDA)");
  m.def("construct_dense_sparse_mapping", &construct_dense_sparse_mapping, "construct mapping from dense to sparse (CUDA)");
  m.def("raycast_occ", &raycast_occ_forward, "raycast_color 3d occupancy grid (CUDA)");
}
