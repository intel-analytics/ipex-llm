//
// Copyright 2016 The BigDL Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// This would makes sure Python is aware there is more than one sub-package within bigdl,
// physically located elsewhere.
// Otherwise there would be module not found error in non-pip's setting as Python would
// only search the first bigdl package and end up finding only one sub-package.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ipex.h>
#include <vector>

#define CL_DMMV_BLOCK_SIZE 32
#define DEQUANTIZATION_BLOCK_SIZE 256

// TODO: support other types later
#define QK4_0 64  // use 64 as llm.cpp use such value
#define QR4_0 2
typedef struct {
    sycl::half d;          // delta
    uint8_t qs[QK4_0 / 2];    // nibbles / quants
} block_q4_0;


template <typename scalar_t>
void dequantize_q4_0(const void * vx, const int ib, const int iqs, scalar_t & v1, scalar_t & v2){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const scalar_t d = x[ib].d;

    const uint8_t vui = x[ib].qs[iqs];

    v1 = vui & 0xF;
    v2 = vui >> 4;

    v1 = (v1 - 8.0f) * d;
    v2 = (v2 - 8.0f) * d;
}


template <typename scalar_t>
void qlinear_xpu_kernel(
        const scalar_t* input,
        const u_int8_t* weight,
        scalar_t* output,
        size_t input_size,
        size_t state_size,
        size_t output_size) {

  // in this function, we support batch size is always 1
  sycl::range global_size(output_size * CL_DMMV_BLOCK_SIZE);
  sycl::range local_size(CL_DMMV_BLOCK_SIZE);
  auto cgf = [&](sycl::handler& handle) {
    sycl::accessor<sycl::half, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        tmp(sycl::range<1>(CL_DMMV_BLOCK_SIZE), handle);
    handle.parallel_for(sycl::nd_range<1>(global_size, local_size),
      [=](sycl::nd_item<1> item) {
          const int row = item.get_group()[0];
          const int tid = item.get_local_id()[0];
          const int y_offset = QK4_0 / QR4_0;
          tmp[tid] = 0;
          for (int i = 0; i < state_size/CL_DMMV_BLOCK_SIZE; i += 2) {
            const int col = i*CL_DMMV_BLOCK_SIZE + 2*tid;
            const int ib = (row * state_size + col)/QK4_0; // block index
            const int iqs = (col % QK4_0)/QR4_0;         // quant index
            const int iybs = col - col%QK4_0;       // y block start index

            // dequantize
            scalar_t v0, v1;
            dequantize_q4_0(weight, ib, iqs, v0, v1);

            // matrix multiplication
            tmp[tid] += v0 * input[iybs + iqs + 0];
            tmp[tid] += v1 * input[iybs + iqs + y_offset];
        }

        // sum up partial sums and write back result
        // barrier(CLK_LOCAL_MEM_FENCE);
        item.barrier(sycl::access::fence_space::local_space);
        for (int s = CL_DMMV_BLOCK_SIZE/2; s>0; s>>=1) {
            if (tid < s) {
                tmp[tid] += tmp[tid + s];
            }
            item.barrier(sycl::access::fence_space::local_space);
        }
        if (tid == 0) {
            output[row] = tmp[0];
        }
      });
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);

  queue.submit(cgf);
}


template <typename scalar_t>
void qlinear_xpu_dequantize_kernel(
        const u_int8_t* weight,
        scalar_t* qweight,
        size_t state_size,
        size_t output_size) {
  // shape of qweight is (output_size, state_size)
  int k = state_size * output_size;
  // define the kernel of de-quantizatation weight on GPU
  // TODO: may further optimize this
  auto cgf_de = [&](sycl::handler& cgh) {
    sycl::stream out(10240, 2560, cgh);
    auto kfn = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
      // int i = it.get_global_id(0); // ith element
      int i = it.get_local_id()[0] * 2 + it.get_group()[0] * it.get_local_range()[0];
      // out << "global id : " << it.get_global_id(0) << " local id" << it.get_local_id()[0] << " , local_range is " <<  it.get_local_range()[0] << ", group is : " << it.get_group()[0] << sycl::endl;
      int ib = i / QK4_0; // Block index
      if(i >= k) return;
      int iqs = (i % QK4_0) / QR4_0; // quantize index inside block
      int iybs = i - i % QK4_0; // y block start
      int y_offset = QK4_0 / QR4_0;
      float v1 = 0;
      float v2 = 0;
      dequantize_q4_0(weight, ib, iqs, v1, v2);
      qweight[iybs + iqs + 0]        = v1;
      qweight[iybs + iqs + y_offset] = v2;
    };
    cgh.parallel_for(
           sycl::nd_range<1>(k,
                             DEQUANTIZATION_BLOCK_SIZE),
           kfn);
  };

  // submit kernel
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);
  queue.submit(cgf_de).wait();
}


torch::Tensor linear_xpu_forward(
        torch::Tensor input,
        torch::Tensor weight) {
  const auto batch_size = input.size(0);
  const auto state_size = input.size(1);
  const long output_size = weight.size(0) / sizeof(block_q4_0) * QK4_0 / state_size;
  
  // output tensor
  auto output_tensor = at::empty({batch_size, output_size}, at::device(at::kXPU).dtype(input.dtype()));

  // below code only handles input dim is (1, N)
  if(batch_size == 1){
    // matrix * vector, use logic similar to OpenCL
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "qlinear_forward_xpu", ([&] {
      qlinear_xpu_kernel<scalar_t>(
            input.data_ptr<scalar_t>(),
            // weight.data_ptr<block_q4_0>(), // quantized weight is u_int8
            weight.data_ptr<u_int8_t>(), // quantized weight is u_int8
            output_tensor.data_ptr<scalar_t>(),
            batch_size, //
            state_size,  // N
            output_size);  // P
    }));

    return output_tensor;
  }
  else{
    // matrix * matrix, just dequantize weight to original format and call linear op
    // Temporary variables to save dequantized fp32 weight
    auto de_weight = at::empty({output_size, state_size}, at::device(at::kXPU).dtype(input.dtype()));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "qlinear_forward_xpu", ([&] {
      qlinear_xpu_dequantize_kernel<scalar_t>(
            weight.data_ptr<u_int8_t>(), // quantized weight is u_int8
            de_weight.data_ptr<scalar_t>(), // temporary variables to save dequantized fp32 weight
            state_size,  // N
            output_size);  // P
    }));
    // just call ipex linear op here
    return at::linear(input, de_weight);
  }
}
