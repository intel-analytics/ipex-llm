#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

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
  size_t *jWeightSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, weightSize, 0));
  size_t *jStrides    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));
  jint *jPads       = (jint*)((*env)->GetPrimitiveArrayCritical(env, pads, 0));

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

  (*env)->ReleasePrimitiveArrayCritical(env, weightSize, jWeightSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides   , jStrides   , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, pads      , jPads      , 0);

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
  size_t *jWeightSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, weightSize, 0));
  size_t *jStrides    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));
  jint *jPads       = (jint*)((*env)->GetPrimitiveArrayCritical(env, pads, 0));

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

  (*env)->ReleasePrimitiveArrayCritical(env, weightSize, jWeightSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides   , jStrides   , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, pads      , jPads      , 0);

  return (jlong)primitive;
}
