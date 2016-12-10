#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "cpu_info.hpp"
#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    poolCreateForward
 * Signature: (IJ[J[J[II)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_poolCreateForward
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong layout,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint borderType)
{
  size_t *jWeightSize = (size_t*)(env->GetPrimitiveArrayCritical(weightSize, 0));
  size_t *jStrides    = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));
  jint *jPads       = (jint*)(env->GetPrimitiveArrayCritical(pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)borderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;
  dnnLayout_t jLayout       = (dnnLayout_t)layout;

  status = dnnPoolingCreateForward_F32(
    &primitive,
    NULL,
    jAlgorithm,
    jLayout,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(weightSize, jWeightSize, 0);
  env->ReleasePrimitiveArrayCritical(strides   , jStrides   , 0);
  env->ReleasePrimitiveArrayCritical(pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    poolCreateBackward
 * Signature: (IJ[J[J[II)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_poolCreateBackward
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong layout,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint borderType)
{
  size_t *jWeightSize = (size_t*)(env->GetPrimitiveArrayCritical(weightSize, 0));
  size_t *jStrides    = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));
  jint *jPads       = (jint*)(env->GetPrimitiveArrayCritical(pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)borderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;
  dnnLayout_t jLayout       = (dnnLayout_t)layout;

  status = dnnPoolingCreateBackward_F32(
    &primitive,
    NULL,
    jAlgorithm,
    jLayout,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(weightSize, jWeightSize, 0);
  env->ReleasePrimitiveArrayCritical(strides   , jStrides   , 0);
  env->ReleasePrimitiveArrayCritical(pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    poolForwardExecute
 * Signature: ([F[F[FJ)J
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_poolForwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray output,
 jfloatArray workspace,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jInput     = (jfloat*)(env->GetPrimitiveArrayCritical(input, 0));
  jfloat* jOutput    = (jfloat*)(env->GetPrimitiveArrayCritical(output, 0));
  jfloat* jWorkspace = (jfloat*)(env->GetPrimitiveArrayCritical(workspace, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc]       = (void*)jInput;
  resources[dnnResourceDst]       = (void*)jOutput;
  resources[dnnResourceWorkspace] = (void*)jWorkspace;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(input    , jInput    , 0);
  env->ReleasePrimitiveArrayCritical(output   , jOutput   , 0);
  env->ReleasePrimitiveArrayCritical(workspace, jWorkspace, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    poolBackwardExecute
 * Signature: ([F[F[FJ)J
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_poolBackwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray gradInput,
 jfloatArray gradOutput,
 jfloatArray workspace,
 jlong primitive)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jGradInput  = (jfloat*)(env->GetPrimitiveArrayCritical(gradInput, 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));
  jfloat* jWorkspace  = (jfloat*)(env->GetPrimitiveArrayCritical(workspace, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceDiffSrc]   = (void*)jGradInput;
  resources[dnnResourceDiffDst]   = (void*)jGradOutput;
  resources[dnnResourceWorkspace] = (void*)jWorkspace;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(gradInput , jGradInput , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(workspace , jWorkspace , 0);
}
