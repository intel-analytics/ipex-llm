#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "cpu_info.hpp"
#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateForwardBias
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateForwardBias
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateForwardBias_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackData
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackData
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardData_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackWeight
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackWeight
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)(env->GetPrimitiveArrayCritical(inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardFilter_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackBias
 * Signature: (J[J)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackBias
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray outputSize)
{
  size_t *jOutputSize  = (size_t*)(env->GetPrimitiveArrayCritical(outputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardBias_F32(
    &primitive,
    NULL,
    dimension,
    jOutputSize);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(outputSize , jOutputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearForwardExecute
 * Signature: ([F[F[F[FJ)J
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearForwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray weight,
 jfloatArray bias,
 jfloatArray output,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

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
 * Method:    linearBackDataExecute
 * Signature: ([F[F[FJ)J
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearBackDataExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray gradInput,
 jfloatArray gradOutput,
 jfloatArray weight,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;
  void *resources[dnnResourceNumber];

  jfloat* jGradInput  = (jfloat*)(env->GetPrimitiveArrayCritical(gradInput , 0));
  jfloat* jWeight     = (jfloat*)(env->GetPrimitiveArrayCritical(weight, 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));

  resources[dnnResourceDiffDst] = jGradOutput;
  resources[dnnResourceFilter]  = jWeight;
  resources[dnnResourceDiffSrc] = jGradInput;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(gradInput , jGradInput , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(weight    , jWeight    , 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearBackWeightExecute
 * Signature: ([F[F[FJ)J
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearBackWeightExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray gradOutput,
 jfloatArray gradWeight,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

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
 * Method:    linearBackBiasExecute
 * Signature: ([F[FJ)J
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearBackBiasExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray gradOutput,
 jfloatArray gradBias,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

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
