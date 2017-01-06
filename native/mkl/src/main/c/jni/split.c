#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    splitCreate
 * Signature: (JJ[J)Ljava/lang/Long;
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_splitCreate
(JNIEnv *env,
 jclass cls,
 jlong numSplits,
 jlong layout,
 jlongArray splitDistri)
{
  dnnPrimitive_t primitive = NULL;
  dnnError_t status = E_UNIMPLEMENTED;
  size_t jNumSplits = (size_t)numSplits;
  dnnLayout_t jLayout = (dnnLayout_t)layout;
  size_t* jSplitDistri = (size_t*)((*env)->GetPrimitiveArrayCritical(env, splitDistri, 0));

  status = dnnSplitCreate_F32(&primitive, NULL, numSplits, jLayout, jSplitDistri);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, splitDistri, jSplitDistri, 0);

  return (long)primitive;
}
