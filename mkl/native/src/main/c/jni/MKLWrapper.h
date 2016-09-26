#ifndef _MKLWARPPER_H
#define _MKLWARPPER_H
#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

template <typename Type>
dnnError_t dnnGroupsConvolutionCreateForwardBias(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateForwardBias_F32(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}
template <>
dnnError_t dnnGroupsConvolutionCreateForwardBias<double>(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateForwardBias_F64(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}

template <typename Type>
dnnError_t dnnGroupsConvolutionCreateBackwardData(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateBackwardData_F32(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}
template <>
dnnError_t dnnGroupsConvolutionCreateBackwardData<double>(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateBackwardData_F64(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}
template <typename Type>
dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateBackwardFilter_F32(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}
template <>
dnnError_t dnnGroupsConvolutionCreateBackwardFilter<double>(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
    const size_t convolutionStrides[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnGroupsConvolutionCreateBackwardFilter_F64(
      pConvolution, attributes, algorithm, groups, dimension, srcSize, dstSize,
      filterSize, convolutionStrides, inputOffset, borderType);
}
template <typename Type>
dnnError_t dnnGroupsConvolutionCreateBackwardBias(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[])
{
  return dnnGroupsConvolutionCreateBackwardBias_F32(
      pConvolution, attributes, algorithm, groups, dimension, dstSize);
}
template <>
dnnError_t dnnGroupsConvolutionCreateBackwardBias<double>(
    dnnPrimitive_t *pConvolution, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm, size_t groups, size_t dimension,
    const size_t dstSize[])
{
  return dnnGroupsConvolutionCreateBackwardBias_F64(
      pConvolution, attributes, algorithm, groups, dimension, dstSize);
}

template <typename Type>
dnnError_t dnnExecute(dnnPrimitive_t primitive, void *resources[])
{
  return dnnExecute_F32(primitive, resources);
}
template <>
dnnError_t dnnExecute<double>(dnnPrimitive_t primitive, void *resources[])
{
  return dnnExecute_F64(primitive, resources);
}

template <typename Type>
dnnError_t dnnReLUCreateForward(dnnPrimitive_t *pRelu,
                                dnnPrimitiveAttributes_t attributes,
                                const dnnLayout_t dataLayout,
                                Type negativeSlope)
{
  return dnnReLUCreateForward_F32(pRelu, attributes, dataLayout, negativeSlope);
}
template <>
dnnError_t dnnReLUCreateForward<double>(dnnPrimitive_t *pRelu,
                                        dnnPrimitiveAttributes_t attributes,
                                        const dnnLayout_t dataLayout,
                                        double negativeSlope)
{
  return dnnReLUCreateForward_F64(pRelu, attributes, dataLayout, negativeSlope);
}
template <typename Type>
dnnError_t dnnReLUCreateBackward(dnnPrimitive_t *pRelu,
                                 dnnPrimitiveAttributes_t attributes,
                                 const dnnLayout_t diffLayout,
                                 const dnnLayout_t dataLayout,
                                 Type negativeSlope)
{
  return dnnReLUCreateBackward_F32(pRelu, attributes, diffLayout, dataLayout,
                                   negativeSlope);
}
template <>
dnnError_t dnnReLUCreateBackward<double>(dnnPrimitive_t *pRelu,
                                         dnnPrimitiveAttributes_t attributes,
                                         const dnnLayout_t diffLayout,
                                         const dnnLayout_t dataLayout,
                                         double negativeSlope)
{
  return dnnReLUCreateBackward_F64(pRelu, attributes, diffLayout, dataLayout,
                                   negativeSlope);
}

template <typename Type>
dnnError_t dnnLayoutCreate(dnnLayout_t *pLayout, size_t dimension,
                           const size_t size[], const size_t strides[])
{
  return dnnLayoutCreate_F32(pLayout, dimension, size, strides);
}

template <>
dnnError_t dnnLayoutCreate<double>(dnnLayout_t *pLayout, size_t dimension,
                                   const size_t size[], const size_t strides[])
{
  return dnnLayoutCreate_F64(pLayout, dimension, size, strides);
}

template <typename Type>
dnnError_t dnnPoolingCreateForward(
    dnnPrimitive_t *pPooling, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op, const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnPoolingCreateForward_F32(pPooling, attributes, op, srcLayout,
                                     kernelSize, kernelStride, inputOffset,
                                     borderType);
}

template <>
dnnError_t dnnPoolingCreateForward<double>(
    dnnPrimitive_t *pPooling, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op, const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnPoolingCreateForward_F64(pPooling, attributes, op, srcLayout,
                                     kernelSize, kernelStride, inputOffset,
                                     borderType);
}

template <typename Type>
dnnError_t dnnPoolingCreateBackward(
    dnnPrimitive_t *pPooling, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op, const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnPoolingCreateBackward_F32(pPooling, attributes, op, srcLayout,
                                      kernelSize, kernelStride, inputOffset,
                                      borderType);
}

template <>
dnnError_t dnnPoolingCreateBackward<double>(
    dnnPrimitive_t *pPooling, dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op, const dnnLayout_t srcLayout, const size_t kernelSize[],
    const size_t kernelStride[], const int inputOffset[],
    const dnnBorder_t borderType)
{
  return dnnPoolingCreateBackward_F64(pPooling, attributes, op, srcLayout,
                                      kernelSize, kernelStride, inputOffset,
                                      borderType);
}

template <typename Type>
dnnError_t dnnLayoutCreateFromPrimitive(dnnLayout_t *pLayout,
                                        const dnnPrimitive_t primitive,
                                        dnnResourceType_t type)
{
  return dnnLayoutCreateFromPrimitive_F32(pLayout, primitive, type);
}

template <>
dnnError_t dnnLayoutCreateFromPrimitive<double>(dnnLayout_t *pLayout,
                                                const dnnPrimitive_t primitive,
                                                dnnResourceType_t type)
{
  return dnnLayoutCreateFromPrimitive_F64(pLayout, primitive, type);
}

template <typename Type>
dnnError_t dnnDelete(dnnPrimitive_t primitive)
{
  return dnnDelete_F32(primitive);
}

template <>
dnnError_t dnnDelete<double>(dnnPrimitive_t primitive)
{
  return dnnDelete_F64(primitive);
}

template <typename Type>
dnnError_t dnnLayoutDelete(dnnLayout_t layout)
{
  return dnnLayoutDelete_F32(layout);
}
template <>
dnnError_t dnnLayoutDelete<double>(dnnLayout_t layout)
{
  return dnnLayoutDelete_F64(layout);
}

template <typename Type>
int dnnLayoutCompare(const dnnLayout_t L1, const dnnLayout_t L2)
{
  return dnnLayoutCompare_F32(L1, L2);
}
template <>
int dnnLayoutCompare<double>(const dnnLayout_t L1, const dnnLayout_t L2)
{
  return dnnLayoutCompare_F64(L1, L2);
}

template <typename Type>
size_t dnnLayoutGetMemorySize(const dnnLayout_t Layout)
{
  return dnnLayoutGetMemorySize_F32(Layout);
}
template <>
size_t dnnLayoutGetMemorySize<double>(const dnnLayout_t Layout)
{
  return dnnLayoutGetMemorySize_F64(Layout);
}

template <typename Type>
dnnError_t dnnAllocateBuffer(void **pPtr, dnnLayout_t layout)
{
  return dnnAllocateBuffer_F32(pPtr, layout);
}
template <>
dnnError_t dnnAllocateBuffer<double>(void **pPtr, dnnLayout_t layout)
{
  return dnnAllocateBuffer_F64(pPtr, layout);
}

template <typename Type>
dnnError_t dnnConversionCreate(dnnPrimitive_t *pConversion,
                               const dnnLayout_t from, const dnnLayout_t to)
{
  return dnnConversionCreate_F32(pConversion, from, to);
}
template <>
dnnError_t dnnConversionCreate<double>(dnnPrimitive_t *pConversion,
                                       const dnnLayout_t from,
                                       const dnnLayout_t to)
{
  return dnnConversionCreate_F64(pConversion, from, to);
}

template <typename Type>
dnnError_t dnnReleaseBuffer(void *pPtr)
{
  return dnnReleaseBuffer_F32(pPtr);
}
template <>
dnnError_t dnnReleaseBuffer<double>(void *pPtr)
{
  return dnnReleaseBuffer_F64(pPtr);
}

template <typename Type>
dnnError_t dnnBatchNormalizationCreateForward(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateForward_F32(pBatchNormalization, attributes,
                                                dataLayout, eps);
}

template <>
dnnError_t dnnBatchNormalizationCreateForward<double>(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateForward_F64(pBatchNormalization, attributes,
                                                dataLayout, eps);
}

template <typename Type>
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateBackwardScaleShift_F32(
      pBatchNormalization, attributes, dataLayout, eps);
}

template <>
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift<double>(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateBackwardScaleShift_F64(
      pBatchNormalization, attributes, dataLayout, eps);
}

template <typename Type>
dnnError_t dnnBatchNormalizationCreateBackwardData(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateBackwardData_F32(
      pBatchNormalization, attributes, dataLayout, eps);
}

template <>
dnnError_t dnnBatchNormalizationCreateBackwardData<double>(
    dnnPrimitive_t *pBatchNormalization, dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout, float eps)
{
  return dnnBatchNormalizationCreateBackwardData_F64(
      pBatchNormalization, attributes, dataLayout, eps);
}

template <typename Type>
dnnError_t dnnLRNCreateForward(dnnPrimitive_t *pLrn,
                               dnnPrimitiveAttributes_t attributes,
                               const dnnLayout_t dataLayout, size_t kernelSie,
                               float alpha, float beta, float k)
{
  return dnnLRNCreateForward_F32(pLrn, attributes, dataLayout, kernelSie, alpha,
                                 beta, k);
}

template <>
dnnError_t dnnLRNCreateForward<double>(dnnPrimitive_t *pLrn,
                                       dnnPrimitiveAttributes_t attributes,
                                       const dnnLayout_t dataLayout,
                                       size_t kernelSie, float alpha,
                                       float beta, float k)
{
  return dnnLRNCreateForward_F64(pLrn, attributes, dataLayout, kernelSie, alpha,
                                 beta, k);
}

template <typename Type>
dnnError_t dnnLRNCreateBackward(dnnPrimitive_t *pLrn,
                                dnnPrimitiveAttributes_t attributes,
                                const dnnLayout_t diffLayout,
                                const dnnLayout_t dataLayout, size_t kernelSize,
                                float alpha, float beta, float k)
{
  return dnnLRNCreateBackward_F32(pLrn, attributes, diffLayout, dataLayout,
                                  kernelSize, alpha, beta, k);
}

template <>
dnnError_t dnnLRNCreateBackward<double>(dnnPrimitive_t *pLrn,
                                        dnnPrimitiveAttributes_t attributes,
                                        const dnnLayout_t diffLayout,
                                        const dnnLayout_t dataLayout,
                                        size_t kernelSize, float alpha,
                                        float beta, float k)
{
  return dnnLRNCreateBackward_F64(pLrn, attributes, diffLayout, dataLayout,
                                  kernelSize, alpha, beta, k);
}

template <typename Type>
dnnError_t dnnInnerProductCreateForwardBias(dnnPrimitive_t *pInnerProduct,
                                            dnnPrimitiveAttributes_t attributes,
                                            size_t dimentions,
                                            const size_t srcSize[],
                                            size_t outputChannels)
{
  return dnnInnerProductCreateForwardBias_F32(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}
template <>
dnnError_t dnnInnerProductCreateForwardBias<double>(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t srcSize[], size_t outputChannels)
{
  return dnnInnerProductCreateForwardBias_F64(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}

template <typename Type>
dnnError_t dnnInnerProductCreateBackwardData(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t srcSize[], size_t outputChannels)
{
  return dnnInnerProductCreateBackwardData_F32(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}
template <>
dnnError_t dnnInnerProductCreateBackwardData<double>(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t srcSize[], size_t outputChannels)
{
  return dnnInnerProductCreateBackwardData_F64(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}
template <typename Type>
dnnError_t dnnInnerProductCreateBackwardFilter(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t srcSize[], size_t outputChannels)
{
  return dnnInnerProductCreateBackwardFilter_F32(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}
template <>
dnnError_t dnnInnerProductCreateBackwardFilter<double>(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t srcSize[], size_t outputChannels)
{
  return dnnInnerProductCreateBackwardFilter_F64(
      pInnerProduct, attributes, dimentions, srcSize, outputChannels);
}
template <typename Type>
dnnError_t dnnInnerProductCreateBackwardBias(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t dstSize[])
{
  return dnnInnerProductCreateBackwardBias_F32(pInnerProduct, attributes,
                                               dimentions, dstSize);
}
template <>
dnnError_t dnnInnerProductCreateBackwardBias<double>(
    dnnPrimitive_t *pInnerProduct, dnnPrimitiveAttributes_t attributes,
    size_t dimentions, const size_t dstSize[])
{
  return dnnInnerProductCreateBackwardBias_F64(pInnerProduct, attributes,
                                               dimentions, dstSize);
}

template <typename Type>
dnnError_t dnnConcatCreate(dnnPrimitive_t *pConcat,
                           dnnPrimitiveAttributes_t attributes,
                           size_t nSrcTensors, dnnLayout_t *src)
{
  return dnnConcatCreate_F32(pConcat, attributes, nSrcTensors, src);
}

template <>
dnnError_t dnnConcatCreate<double>(dnnPrimitive_t *pConcat,
                                   dnnPrimitiveAttributes_t attributes,
                                   size_t nSrcTensors, dnnLayout_t *src)
{
  return dnnConcatCreate_F64(pConcat, attributes, nSrcTensors, src);
}

template <typename Type>
dnnError_t dnnSplitCreate(dnnPrimitive_t *pSplit,
                          dnnPrimitiveAttributes_t attributes,
                          const size_t nDstTensors, dnnLayout_t layout,
                          size_t dstChannelSize[])
{
  
  return dnnSplitCreate_F32(pSplit, attributes, nDstTensors, layout,
                            dstChannelSize);
}

template <>
dnnError_t dnnSplitCreate<double>(dnnPrimitive_t *pSplit,
                                  dnnPrimitiveAttributes_t attributes,
                                  const size_t nDstTensors, dnnLayout_t layout,
                                  size_t dstChannelSize[])
{
  
  return dnnSplitCreate_F64(pSplit, attributes, nDstTensors, layout,
                            dstChannelSize);
}

template <typename Type>
dnnError_t dnnSumCreate(
  dnnPrimitive_t *pSum,
  dnnPrimitiveAttributes_t attributes, const size_t nSummands,
  dnnLayout_t layout, Type *coefficients)
{
  return dnnSumCreate_F32(pSum, attributes, nSummands, layout, coefficients);
}

template <>
dnnError_t dnnSumCreate<double>(
  dnnPrimitive_t *pSum,
  dnnPrimitiveAttributes_t attributes, const size_t nSummands,
  dnnLayout_t layout, double *coefficients)
{
  return dnnSumCreate_F64(pSum, attributes, nSummands, layout, coefficients);
}
#endif
