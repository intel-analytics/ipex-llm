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

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    reluForwardExecute
 * Signature: ([F[FJ)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_reluForwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray output,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jInput  = (jfloat*)(env->GetPrimitiveArrayCritical(input, 0));
  jfloat* jOutput = (jfloat*)(env->GetPrimitiveArrayCritical(output, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc]    = (void*)jInput;
  resources[dnnResourceDst]    = (void*)jOutput;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(input , jInput , 0);
  env->ReleasePrimitiveArrayCritical(output, jOutput, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    reluBackwardExecute
 * Signature: ([F[F[FJ)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_reluBackwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray gradInput,
 jfloatArray gradOutput,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;
  void *resources[dnnResourceNumber];

  jfloat* jInput      = (jfloat*)(env->GetPrimitiveArrayCritical(input, 0));
  jfloat* jGradInput  = (jfloat*)(env->GetPrimitiveArrayCritical(gradInput , 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));

  resources[dnnResourceSrc]     = (void*)jInput;
  resources[dnnResourceDiffDst] = (void*)jGradOutput;
  resources[dnnResourceDiffSrc] = (void*)jGradInput;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(gradInput , jGradInput , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(input     , jInput     , 0);
}
