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

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    lrnForwardExecute
 * Signature: ([F[F[FJ)V
 */
  JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_lrnForwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray output,
 jfloatArray workspace,
 jlong primitive)
{
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
 * Method:    lrnBackwardExecute
 * Signature: ([F[F[F[FJ)V
 */
  JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_lrnBackwardExecute
(JNIEnv *env,
 jclass cls,
 jfloatArray input,
 jfloatArray gradInput,
 jfloatArray gradOutput,
 jfloatArray workspace,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jInput      = (jfloat*)(env->GetPrimitiveArrayCritical(input    , 0));
  jfloat* jGradInput  = (jfloat*)(env->GetPrimitiveArrayCritical(gradInput, 0));
  jfloat* jGradOutput = (jfloat*)(env->GetPrimitiveArrayCritical(gradOutput, 0));
  jfloat* jWorkspace  = (jfloat*)(env->GetPrimitiveArrayCritical(workspace, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceSrc]       = (void*)jInput;
  resources[dnnResourceDiffSrc]   = (void*)jGradInput;
  resources[dnnResourceDiffDst]   = (void*)jGradOutput;
  resources[dnnResourceWorkspace] = (void*)jWorkspace;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(input     , jInput     , 0);
  env->ReleasePrimitiveArrayCritical(gradInput , jGradInput , 0);
  env->ReleasePrimitiveArrayCritical(gradOutput, jGradOutput, 0);
  env->ReleasePrimitiveArrayCritical(workspace , jWorkspace , 0);
}
