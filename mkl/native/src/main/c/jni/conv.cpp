#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"
/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateForward
 * Signature: (III[I[I[I[I[II)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateForward
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));
  size_t *jOutputSize = (size_t*)(env->GetPrimitiveArrayCritical(outputSize, 0));
  size_t *jWeightSize = (size_t*)(env->GetPrimitiveArrayCritical(weightSize, 0));
  size_t *jStrides    = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));
  jint *jPads       = (jint*)(env->GetPrimitiveArrayCritical(pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateForwardBias_F32(
    &primitive,
    NULL,
    dnnAlgorithmConvolutionDirect,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    dnnBorderZeros);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);
  env->ReleasePrimitiveArrayCritical(outputSize, jOutputSize, 0);
  env->ReleasePrimitiveArrayCritical(weightSize, jWeightSize, 0);
  env->ReleasePrimitiveArrayCritical(strides   , jStrides   , 0);
  env->ReleasePrimitiveArrayCritical(pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionForwardExecute
 * Signature: ([F[F[F[FJ)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionForwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray weight,
 jfloatArray bias,
 jfloatArray output,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jInput  = (jfloat*)(env->GetPrimitiveArrayCritical(input, 0));
  jfloat* jWeight = (jfloat*)(env->GetPrimitiveArrayCritical(weight, 0));
  jfloat* jBias   = (jfloat*)(env->GetPrimitiveArrayCritical(bias, 0));
  jfloat* jOutput = (jfloat*)(env->GetPrimitiveArrayCritical(output, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceFilter] = (void*)jWeight;
  resources[dnnResourceBias]   = (void*)jBias;
  resources[dnnResourceSrc]    = (void*)jInput;
  resources[dnnResourceDst]    = (void*)jOutput;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(input , jInput , 0);
  env->ReleasePrimitiveArrayCritical(weight, jWeight, 0);
  env->ReleasePrimitiveArrayCritical(bias  , jBias  , 0);
  env->ReleasePrimitiveArrayCritical(output, jOutput, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardData
 * Signature: (IJJ[J[J[J[J[II)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardData
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));
  size_t *jOutputSize = (size_t*)(env->GetPrimitiveArrayCritical(outputSize, 0));
  size_t *jWeightSize = (size_t*)(env->GetPrimitiveArrayCritical(weightSize, 0));
  size_t *jStrides    = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));
  jint   *jPads       = (jint*)(env->GetPrimitiveArrayCritical(pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateBackwardData_F32(
    &primitive,
    NULL,
    jAlgorithm,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);
  env->ReleasePrimitiveArrayCritical(outputSize, jOutputSize, 0);
  env->ReleasePrimitiveArrayCritical(weightSize, jWeightSize, 0);
  env->ReleasePrimitiveArrayCritical(strides   , jStrides   , 0);
  env->ReleasePrimitiveArrayCritical(pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionBackwardDataExecute
 * Signature: ([F[F[FJ)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionBackwardDataExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray gradInput,
 jfloatArray gradOutput,
 jfloatArray backWeight,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;
  void *resources[dnnResourceNumber];

  jfloat* jGradInput  = (jfloat*)(env->GetPrimitiveArrayCritical(gradInput , 0));
  jfloat* jWeight     = (jfloat*)(env->GetPrimitiveArrayCritical(backWeight, 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));

  resources[dnnResourceDiffDst] = jGradOutput;
  resources[dnnResourceFilter]  = jWeight;
  resources[dnnResourceDiffSrc] = jGradInput;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(gradInput , jGradInput , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(backWeight, jWeight    , 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardKernel
 * Signature: (IJJ[J[J[J[J[II)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardKernel
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));
  size_t *jOutputSize = (size_t*)(env->GetPrimitiveArrayCritical(outputSize, 0));
  size_t *jWeightSize = (size_t*)(env->GetPrimitiveArrayCritical(weightSize, 0));
  size_t *jStrides    = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));
  jint   *jPads       = (jint*)(env->GetPrimitiveArrayCritical(pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateBackwardFilter_F32(
    &primitive,
    NULL,
    jAlgorithm,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);
  env->ReleasePrimitiveArrayCritical(outputSize, jOutputSize, 0);
  env->ReleasePrimitiveArrayCritical(weightSize, jWeightSize, 0);
  env->ReleasePrimitiveArrayCritical(strides   , jStrides   , 0);
  env->ReleasePrimitiveArrayCritical(pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionBackwardKernelExecute
 * Signature: ([F[F[FJ)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionBackwardKernelExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray gradOutput,
 jfloatArray gradWeight,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;
  void *resources[dnnResourceNumber];

  jfloat* jInput      = (jfloat*)(env->GetPrimitiveArrayCritical(input, 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));
  jfloat* jGradWeight = (jfloat*)(env->GetPrimitiveArrayCritical(gradWeight, 0));

  resources[dnnResourceDiffDst]    = jGradOutput;
  resources[dnnResourceSrc]        = jInput;
  resources[dnnResourceDiffFilter] = jGradWeight;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(input     , jInput     , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(gradWeight, jGradWeight, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardBias
 * Signature: (IJJ[J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardBias
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray outputSize)
{
  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  size_t jGroups            = (size_t)groups;
  size_t jDim               = (size_t)dim;
  size_t *jOutputSize       = (size_t*)(env->GetPrimitiveArrayCritical(outputSize, 0));

  dnnError_t status        = E_UNIMPLEMENTED;
  dnnPrimitive_t primitive = NULL;

  status = dnnGroupsConvolutionCreateBackwardBias_F32(
    &primitive,
    NULL,
    jAlgorithm,
    jGroups,
    jDim,
    jOutputSize);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(outputSize, jOutputSize, 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionBackwardBiasExecute
 * Signature: ([F[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionBackwardBiasExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray gradOutput,
 jfloatArray gradBias,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;
  void *resources[dnnResourceNumber];

  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));
  jfloat* jGradBias   = (jfloat*)(env->GetPrimitiveArrayCritical(gradBias, 0));

  resources[dnnResourceDiffDst]  = jGradOutput;
  resources[dnnResourceDiffBias] = jGradBias;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(gradBias  , jGradBias  , 0);
}
