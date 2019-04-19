#include <torch/extension.h>
#include <c10/util/ArrayRef.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>

#include <iostream>
#include <vector>
#include <cassert>


using IntArrayRef = c10::ArrayRef<int64_t>;
using at::native::getCudnnHandle;

at::Tensor dfxp_quantize_forward(
    at::Tensor input,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_) {

    float mx = input.max().item().to<float>();
    float mn = input.min().item().to<float>();
    float qmin = qmin_.item().to<float>();
    float qmax = qmax_.item().to<float>();
    float step = step_.item().to<float>();
    if (mx > qmax * step || mn < qmin * step) {
        step_.mul_(2);
    } else if (mx <= qmax * step / 2 && mn >= qmin * step / 2) {
        step_.div_(2);
    }
    auto x = input.clone();
    x.div_(step).clamp_(qmin, qmax).round_().mul_(step);
    return x;
}

at::Tensor dfxp_stochastic_quantize_forward(
    at::Tensor input,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_) {

    float mx = input.max().item().to<float>();
    float mn = input.min().item().to<float>();
    float qmin = qmin_.item().to<float>();
    float qmax = qmax_.item().to<float>();
    float step = step_.item().to<float>();
    if (mx > qmax * step || mn < qmin * step) {
        step_.mul_(2);
    } else if (mx <= qmax * step / 2 && mn >= qmin * step / 2) {
        step_.div_(2);
    }
    auto x = input.clone();
    auto noise = torch::rand(x.numel(), x.device());
    noise = noise.reshape_as(x);
    x.div_(step).add_(noise);
    x.clamp_(qmin, qmax).round_().mul_(step);
    return x;
}

at::Tensor dfxp_grad_quantize_backward(
    at::Tensor grad,
    at::Tensor qmin_,
    at::Tensor qmax_,
    at::Tensor step_) {

    float mx = grad.max().item().to<float>();
    float mn = grad.min().item().to<float>();
    float qmin = qmin_.item().to<float>();
    float qmax = qmax_.item().to<float>();
    float step = step_.item().to<float>();
    if (mx > qmax * step || mn < qmin * step) {
        step_.mul_(2);
    } else if (mx <= qmax * step / 2 && mn >= qmin * step / 2) {
        step_.div_(2);
    }
    grad.div_(step);
    auto noise = torch::rand(grad.numel(), grad.device());
    noise = noise.reshape_as(grad);
    grad.add_(noise).clamp_(qmin, qmax).floor_().mul_(step);
    return grad;
}

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

    auto dim = input_size.size();
    std::vector<int64_t> output_size(dim);
    output_size[0] = input_size[0];
    output_size[1] = weight_size[0];
    for (size_t d=2; d<dim; ++d) {
        auto kernel = dilation[d-2] * (weight_size[d] - 1) + 1;
        output_size[d] = (input_size[d] + 2 * padding[d-2] - kernel) / stride[d-2] + 1;
    }
    return output_size;
}

at::Tensor dfxp_8bit_convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

    // check data types
    assert(input.scalar_type() == at::kChar);
    assert(weight.scalar_type() == at::kChar);

    // cudnn handle
    auto cudnn = getCudnnHandle();

    auto& x = input;
    auto& w = weight;

    // input desc
    cudnnTensorDescriptor_t idesc;
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        idesc,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_INT8,
        /*batch_size=*/x.size(0),
        /*channels=*/x.size(3),
        /*height=*/x.size(1),
        /*width=*/x.size(2)));

    // weight desc
    cudnnFilterDescriptor_t wdesc;
    AT_CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
    AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        wdesc,
        /*dataType=*/CUDNN_DATA_INT8,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*out_channels=*/w.size(0),
        /*in_channels=*/w.size(3),
        /*kernel_height=*/w.size(1),
        /*kernel_width=*/w.size(2)));

    // convolution desc
    cudnnConvolutionDescriptor_t cdesc;
    AT_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cdesc));
    AT_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        cdesc,
        /*pad_height=*/padding[0],
        /*pad_width=*/padding[1],
        /*vertical_stride=*/stride[0],
        /*horizontal_stride=*/stride[1],
        /*dilation_height=*/dilation[0],
        /*dilation_width=*/dilation[1],
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_INT32));

    // allocate output tensor
    int output_n, output_c, output_h, output_w;
    AT_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        cdesc,
        idesc,
        wdesc,
        /*n=*/&output_n,
        /*c=*/&output_c,
        /*h=*/&output_h,
        /*w=*/&output_w));
    std::vector<int64_t> output_size = {output_n, output_h, output_w, output_c};
    auto output = at::empty(output_size, input.options().dtype(at::kChar));

    // output desc
    cudnnTensorDescriptor_t odesc;
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&odesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        odesc,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_INT8,
        /*batch_size=*/output.size(0),
        /*channels=*/output.size(3),
        /*height=*/output.size(1),
        /*width=*/output.size(2)));

    // std::cerr << x.size(0) << ' '
    //           << x.size(1) << ' '
    //           << x.size(2) << ' '
    //           << x.size(3) << ' '
    //           << std::endl;

    // std::cerr << w.size(0) << ' '
    //           << w.size(1) << ' '
    //           << w.size(2) << ' '
    //           << w.size(3) << ' '
    //           << std::endl;

    // std::cerr << output.size(0) << ' '
    //           << output.size(1) << ' '
    //           << output.size(2) << ' '
    //           << output.size(3) << ' '
    //           << std::endl;

    // std::cerr << CUDNN_MAJOR << '.' << CUDNN_MINOR << std::endl;

    // convolution algo
    cudnnConvolutionFwdAlgo_t conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // AT_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
    //     cudnn,
    //     idesc,
    //     wdesc,
    //     cdesc,
    //     odesc,
    //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //     /*memoryLimitInBytes=*/0,
    //     &conv_algo));

    // allocate workspace
    size_t workspace_bytes = 0;
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        idesc,
        wdesc,
        cdesc,
        odesc,
        conv_algo,
        &workspace_bytes));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // convolution
    const float alpha = 1, beta = 0;
    AT_CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha,
        idesc,
        input.data_ptr(),
        wdesc,
        w.data_ptr(),
        cdesc,
        conv_algo,
        d_workspace,
        workspace_bytes,
        &beta,
        odesc,
        output.data_ptr()));

    // clean up
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(idesc);
    cudnnDestroyTensorDescriptor(odesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyConvolutionDescriptor(cdesc);

    // perm = {0, 3, 1, 2};
    // return output.permute(perm).contiguous();
    return output;
}

at::Tensor dfxp_32bit_convolution_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

    // check data types
    assert(input.scalar_type() == at::kFloat);
    assert(weight.scalar_type() == at::kFloat);

    // cudnn handle
    auto cudnn = getCudnnHandle();

    // input desc
    cudnnTensorDescriptor_t idesc;
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&idesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        idesc,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/input.size(0),
        /*channels=*/input.size(1),
        /*height=*/input.size(2),
        /*width=*/input.size(3)));

    // weight desc
    auto w = weight.contiguous(); // contiguous version of weight
    cudnnFilterDescriptor_t wdesc;
    AT_CUDNN_CHECK(cudnnCreateFilterDescriptor(&wdesc));
    AT_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        wdesc,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/w.size(0),
        /*in_channels=*/w.size(1),
        /*kernel_height=*/w.size(2),
        /*kernel_width=*/w.size(3)));

    // std::cerr << w.size(0) << ' '
    //           << w.size(1) << ' '
    //           << w.size(2) << ' '
    //           << w.size(3) << ' '
    //           << std::endl;

    // allocate output tensor
    auto output = at::empty(conv_output_size(
        input.sizes(),
        w.sizes(),
        padding,
        stride,
        dilation,
        groups), input.options().dtype(at::kFloat));

    // output desc
    cudnnTensorDescriptor_t odesc;
    AT_CUDNN_CHECK(cudnnCreateTensorDescriptor(&odesc));
    AT_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        odesc,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/output.size(0),
        /*channels=*/output.size(1),
        /*height=*/output.size(2),
        /*width=*/output.size(3)));

    // convolution desc
    cudnnConvolutionDescriptor_t cdesc;
    AT_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cdesc));
    AT_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        cdesc,
        /*pad_height=*/padding[0],
        /*pad_width=*/padding[1],
        /*vertical_stride=*/stride[0],
        /*horizontal_stride=*/stride[1],
        /*dilation_height=*/dilation[0],
        /*dilation_width=*/dilation[1],
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));

    // convolution algo
    cudnnConvolutionFwdAlgo_t conv_algo;
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        idesc,
        wdesc,
        cdesc,
        odesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        &conv_algo));

    // allocate workspace
    size_t workspace_bytes = 0;
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        idesc,
        wdesc,
        cdesc,
        odesc,
        conv_algo,
        &workspace_bytes));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspace_bytes);

    // convolution
    const float alpha = 1, beta = 0;
    AT_CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha,
        idesc,
        input.data_ptr(),
        wdesc,
        w.data_ptr(),
        cdesc,
        conv_algo,
        d_workspace,
        workspace_bytes,
        &beta,
        odesc,
        output.data_ptr()));

    // clean up
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(idesc);
    cudnnDestroyTensorDescriptor(odesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyConvolutionDescriptor(cdesc);

    return output;
}

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <ctime>
#include <cfloat>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "cuda.h"
#include "cudnn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

/** Convolutional layer */
struct ConvolutionLayer {
  int kernel_size;
  int in_channels, in_height, in_width;
  int out_channels, out_height, out_width;
  std::vector<float> pconv;

  ConvolutionLayer(int in_channels_,
                   int out_channels_,
                   int kernel_size_,
                   int in_w_,
                   int in_h_)
    : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;
  }
};

/** Training context */
struct TrainingContext {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;
  cudnnFilterDescriptor_t conv1filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc;
  cudnnConvolutionFwdAlgo_t conv1algo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;

  // Disable copying
  TrainingContext& operator=(const TrainingContext&) = delete;
  TrainingContext(const TrainingContext&) = delete;

  // Constructor
  TrainingContext(int gpuid, int batch_size, ConvolutionLayer& conv1)
    : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    /** Create descriptors within the constructor.
      * As instructed in the Usual manual, descriptors for
      * input and output tensors, filter, and the forward
      * convolution operator are created along with
      * cuDNN handle.
      */
    printf("set cuda device\n");
    checkCudaErrors(cudaSetDevice(gpuid));
    printf("creating cudnn handle\n");
    checkCUDNN(cudnnCreate(&cudnnHandle));
    printf("creating descriptors\n");
    checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
    checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
    checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));

    // Initialize convolution forward pass
    printf("calculating workspace size\n");
    size_t workspaceSizeFromConv = SetFwdConvolutionTensors(
        conv1, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo);
    m_workspaceSize = std::max((int)workspaceSizeFromConv, 0);
  }

  ~TrainingContext() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnDestroy(cudnnHandle));

    checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
    checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
    checkCUDNN(cudnnDestroyFilterDescriptor(conv1filterDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(conv1Desc));
  }

  /** Set tensors and ops for forward pass */
  size_t SetFwdConvolutionTensors(ConvolutionLayer& conv,
                                  cudnnTensorDescriptor_t& srcTensorDesc,
                                  cudnnTensorDescriptor_t& dstTensorDesc,
                                  cudnnFilterDescriptor_t& filterDesc,
                                  cudnnConvolutionDescriptor_t& convDesc,
                                  cudnnConvolutionFwdAlgo_t& algo) {
    int n = m_batchSize;
    int c = conv.in_channels;
    int h = conv.in_height;
    int w = conv.in_width;

    // Set input tensor. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetTensor4dDescriptor(
        srcTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, n, c, h, w));

    // Set convolution filter. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                          CUDNN_DATA_INT8,
                                          CUDNN_TENSOR_NHWC,
                                          conv.out_channels,
                                          conv.in_channels,
                                          conv.kernel_size,
                                          conv.kernel_size));

    // Set convolution operator. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT32
    int pad_height = 0;
    int pad_width = 0;
    int stride_h = 1;
    int stride_v = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                               pad_height,
                                               pad_width,
                                               stride_h,
                                               stride_v,
                                               dilation_h,
                                               dilation_w,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_DATA_INT32));

    // Compute output dimension. Folowing the manual, chagnged
    // * CUDNN_DATA_FLOAT -> CUDNN_DATA_INT8, and 
    // * CUDNN_TENSOR_NCHW -> CUDNN_TENSOR_NHWC
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));

    // Set output tensor (Changed CUDNN_DATA_FLOAT to CUDNN_DATA_INT8, following the manual)
    checkCUDNN(cudnnSetTensor4dDescriptor(
        dstTensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, n, c, h, w));

    // Retrieve orward pass algorithm. We can either hardcode it to a specific
    // algorithm or use cudnnGetConvolutionForwardAlgorithm. For the purpose
    // of this test, either way works.
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // Following also works
    // checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
    //     cudnnHandle,
    //     srcTensorDesc,
    //     filterDesc,
    //     convDesc,
    //     dstTensorDesc,
    //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //     0,
    //     &algo));
    

    // Compute workspace size. We can either hardcode it to a specific number,
    // or use cudnnGetConvolutionForwardWorkspaceSize. For the purpose of this
    // test, either way works.
    size_t sizeInBytes = 1073741824;    
    // Following also works
    // size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       algo,
                                                       &sizeInBytes));
    

    return sizeInBytes;
  }

  /** Execute forward pass */
  void ForwardPropagation(float* data,
                          float* conv1,
                          float* pconv1,
                          void* workspace) {
    float alpha = 1.0f;
    float beta = 0.0f;
    checkCudaErrors(cudaSetDevice(m_gpuid));
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       dataTensor,
                                       data,
                                       conv1filterDesc,
                                       pconv1,
                                       conv1Desc,
                                       conv1algo,
                                       workspace,
                                       m_workspaceSize,
                                       &beta,
                                       conv1Tensor,
                                       conv1));
  }
};

int foo() {
  printf("Start\n");
  printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
  // parameters
  int gpu = 0;
  int iterations = 10000;

  // input dimensions
  size_t width = 960;
  size_t height = 600;
  size_t channels = 4;
  int batch_size = 1;

  // Create layer architecture
  printf("Creating conv layer\n");
  int out_channels = 4;
  int kernel_size = 3;
  ConvolutionLayer conv1(
      (int)channels, out_channels, kernel_size, (int)width, (int)height);

  printf("Creating context\n");
  TrainingContext context(gpu, batch_size, conv1);

  // Initizlie convolution weight
  printf("Filling conv\n");
  std::mt19937 g(42);
  float wconv1 =
      sqrt(3.0f / (conv1.kernel_size * conv1.kernel_size * conv1.in_channels));
  std::uniform_real_distribution<> dconv1(-wconv1, wconv1);
  for (auto&& iter : conv1.pconv) {
    iter = static_cast<float>(dconv1(g));
  }

  // Initailize input image (batch size = 1)
  std::vector<float> img_float(1 * width * height * channels);
  for (auto&& iter : img_float) {
    iter = static_cast<float>(dconv1(g));
  }

  // Allocate input and output on GPU; copy input over to GPU
  float* d_data, *d_conv1;
  checkCudaErrors(cudaMalloc(&d_data,
                             sizeof(float) * context.m_batchSize * channels *
                                 height * width));
  checkCudaErrors(cudaMalloc(&d_conv1,
                             sizeof(float) * context.m_batchSize *
                                 conv1.out_channels * conv1.out_height *
                                 conv1.out_width));
  checkCudaErrors(cudaMemcpyAsync(d_data,
                                  &img_float[0],
                                  sizeof(float) * 1 * channels * width * height,
                                  cudaMemcpyHostToDevice));

  // Allocate kernel on GPU
  float* d_pconv1;
  checkCudaErrors(cudaMalloc(&d_pconv1, sizeof(float) * conv1.pconv.size()));
  checkCudaErrors(cudaMemcpyAsync(d_pconv1,
                                  &conv1.pconv[0],
                                  sizeof(float) * conv1.pconv.size(),
                                  cudaMemcpyHostToDevice));

  // Temporary buffers and workspaces
  void* d_cudnn_workspace = nullptr;
  if (context.m_workspaceSize > 0) {
    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, context.m_workspaceSize));
  }

  // Start forward pass
  printf("Begin forwrad pass\n");
  checkCudaErrors(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterations; ++iter) {
    context.ForwardPropagation(d_data, d_conv1, d_pconv1, d_cudnn_workspace);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();

  printf(
      "Iteration time: %f ms\n",
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
          1000.0f / iterations);

  // Free data structures
  checkCudaErrors(cudaFree(d_data));
  checkCudaErrors(cudaFree(d_conv1));
  checkCudaErrors(cudaFree(d_pconv1));

  if (d_cudnn_workspace != nullptr)
    checkCudaErrors(cudaFree(d_cudnn_workspace));

  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dfxp_quantize_forward", &dfxp_quantize_forward, "DFXP quantize forward");
    m.def("dfxp_stochastic_quantize_forward", &dfxp_stochastic_quantize_forward, "DFXP stochastic quantize forward");
    m.def("dfxp_grad_quantize_backward", &dfxp_grad_quantize_backward, "DFXP grad quantize backward");
    m.def("dfxp_8bit_convolution_forward", &dfxp_8bit_convolution_forward, "DFXP 8bit convolution forward");
    m.def("dfxp_32bit_convolution_forward", &dfxp_32bit_convolution_forward, "DFXP 32bit convolution forward");
    m.def("foo", &foo, "Foo");
}
