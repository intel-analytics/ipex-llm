#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    lrnCreateForward
 * Signature: (JJFFF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_lrnCreateForward
(JNIEnv *env,
 jclass cls,
 jlong layout,
 jlong size,
 jfloat alpha,
 jfloat beta,
 jfloat k)
{
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;
  dnnLayout_t jLayout       = (dnnLayout_t)layout;
  size_t jSize              = (size_t)size;

  status = dnnLRNCreateForward_F32(
    &primitive,
    NULL,
    jLayout,
    jSize,
    alpha,
    beta,
    k);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    lrnCreateBackward
 * Signature: (JJJFFF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_lrnCreateBackward
(JNIEnv *env,
 jclass cls,
 jlong layout1,
 jlong layout2,
 jlong size,
 jfloat alpha,
 jfloat beta,
 jfloat k)
{
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;
  dnnLayout_t jLayout1      = (dnnLayout_t)layout1;
  dnnLayout_t jLayout2      = (dnnLayout_t)layout2;
  size_t              jSize = (size_t)size;

  status = dnnLRNCreateBackward_F32(
    &primitive,
    NULL,
    jLayout1,
    jLayout2,
    jSize,
    alpha,
    beta,
    k);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}
