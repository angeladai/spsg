#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void bilateral_filter_floatmap_cuda(
    at::Tensor output,
    at::Tensor input,
	float sigmaD,
	float sigmaR);

void median_fill_depthmap_cuda(
    at::Tensor output,
    at::Tensor input);

void convert_depth_to_cameraspace_cuda(
    at::Tensor output,
    at::Tensor input,
	at::Tensor intrinsics,
	float depthMin,
	float depthMax);

void compute_normals_cuda(
    at::Tensor output,
    at::Tensor input);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void bilateral_filter_floatmap(
    at::Tensor output,
    at::Tensor input,
	float sigmaD,
	float sigmaR) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);

  return bilateral_filter_floatmap_cuda(output, input, sigmaD, sigmaR);
}

void median_fill_depthmap(
    at::Tensor output,
    at::Tensor input) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);

  return median_fill_depthmap_cuda(output, input);
}

// input: batch x 1 x height x width x 3 (campos)
void convert_depth_to_cameraspace(
    at::Tensor output,
    at::Tensor input,
	at::Tensor intrinsics,
	float depthMin,
	float depthMax) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);
  CHECK_INPUT(intrinsics);

  return convert_depth_to_cameraspace_cuda(output, input, intrinsics, depthMin, depthMax);
}

// input: batch x 1 x height x width x 3 (campos)
void compute_normals(
    at::Tensor output,
    at::Tensor input) {
  CHECK_INPUT(output);
  CHECK_INPUT(input);

  return compute_normals_cuda(output, input);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilateral_filter_floatmap", &bilateral_filter_floatmap, "bilateral filter float map (CUDA)");
  m.def("median_fill_depthmap", &median_fill_depthmap, "median fill depth map (CUDA)");
  m.def("convert_depth_to_cameraspace", &convert_depth_to_cameraspace, "compute camera positions from depth map (CUDA)");
  m.def("compute_normals", &compute_normals, "compute normals from camera positions (CUDA)");
}
