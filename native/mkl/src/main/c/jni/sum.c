#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    sumCreate
 * Signature: (JJ[J)Ljava/lang/Long;
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_sumCreate
(JNIEnv *env,
 jclass cls,
 jlong numSums,
 jlong layout,
 jfloatArray coefficients)
{
  dnnPrimitive_t primitive = NULL;
  jfloat* jCoefficients = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, coefficients, 0));
  size_t jNumSums = (size_t)numSums;
  dnnLayout_t jLayout = (dnnLayout_t)layout;
  dnnError_t status = E_UNIMPLEMENTED;

  status = dnnSumCreate_F32(&primitive, NULL, jNumSums, jLayout, jCoefficients);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, coefficients, jCoefficients, 0);

  return (long)primitive;
}
