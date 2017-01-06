#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"
/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    reluCreateForward
 * Signature: (JF)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_reluCreateForward
(JNIEnv *env,
 jclass cls,
 jlong layout,
 jfloat nagtiveSlope)
{
  dnnLayout_t jLayout      = (dnnLayout_t)layout;
  dnnPrimitive_t primitive = NULL;
  dnnError_t status        = E_UNIMPLEMENTED;

  status = dnnReLUCreateForward_F32(&primitive, NULL, jLayout, nagtiveSlope);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    reluCreateBackward
 * Signature: (JJF)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_reluCreateBackward
(JNIEnv *env,
 jclass cls,
 jlong layout1,
 jlong layout2,
 jfloat nagtiveSlope)
{
  dnnLayout_t jLayout1     = (dnnLayout_t)layout1;
  dnnLayout_t jLayout2     = (dnnLayout_t)layout2;
  dnnPrimitive_t primitive = NULL;
  dnnError_t status        = E_UNIMPLEMENTED;

  status = dnnReLUCreateBackward_F32(&primitive, NULL, jLayout1, jLayout2,
                                     nagtiveSlope);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}
