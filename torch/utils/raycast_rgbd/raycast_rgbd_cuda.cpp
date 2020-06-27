#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void raycast_rgbd_cuda_forward( // TODO: dense or sparse here?
    at::Tensor sparse_mapping,
    at::Tensor locs,
	at::Tensor vals_sdf,
	at::Tensor vals_color,
	at::Tensor vals_normals,
    at::Tensor viewMatrixInv,
    at::Tensor imageColor,
    at::Tensor imageDepth,
    at::Tensor imageNormal,
	at::Tensor mapping2dto3d,
	at::Tensor mapping3dto2d_num,
	at::Tensor intrinsicParams,
    at::Tensor opts);

void raycast_rgbd_cuda_backward(
    at::Tensor grad_color,
    at::Tensor grad_depth,
	at::Tensor grad_normal,
    at::Tensor sparse_mapping,
    at::Tensor mapping3dto2d,
	at::Tensor mapping3dto2d_num,
	at::Tensor dims,
	at::Tensor d_color,
	at::Tensor d_depth,
	at::Tensor d_normals);

void construct_dense_sparse_mapping_cuda(
    at::Tensor locs,
    at::Tensor sparse_mapping);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void raycast_color_forward(
    at::Tensor sparse_mapping,
    at::Tensor locs,
	at::Tensor vals_sdf,
	at::Tensor vals_color,
	at::Tensor vals_normals,
    at::Tensor viewMatrixInv,
    at::Tensor imageColor,
    at::Tensor imageDepth,
    at::Tensor imageNormal,
	at::Tensor mapping3dto2d,
	at::Tensor mapping3dto2d_num,
	at::Tensor intrinsicParams,
    at::Tensor opts) {
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
    at::Tensor locs,
    at::Tensor sparse_mapping) {
  CHECK_INPUT(locs);
  CHECK_INPUT(sparse_mapping);

  return construct_dense_sparse_mapping_cuda(locs, sparse_mapping);
}

void raycast_color_backward(
    at::Tensor grad_color,
	at::Tensor grad_depth,
	at::Tensor grad_normal,
    at::Tensor sparse_mapping,
    at::Tensor mapping3dto2d,
	at::Tensor mapping3dto2d_num,
	at::Tensor dims,
	at::Tensor d_color,
	at::Tensor d_depth,
	at::Tensor d_normals) {
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &raycast_color_forward, "raycast_color forward (CUDA)");
  m.def("backward", &raycast_color_backward, "raycast_color backward (CUDA)");
  m.def("construct_dense_sparse_mapping", &construct_dense_sparse_mapping, "construct mapping from dense to sparse (CUDA)");
}
