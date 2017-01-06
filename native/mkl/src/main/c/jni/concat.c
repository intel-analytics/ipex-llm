#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    concatCreate
 * Signature: (J[J)Ljava/lang/Long;
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_concatCreate
(JNIEnv *env,
 jclass cls,
 jlong numConcats,
 jlongArray layouts)
{
  dnnLayout_t* jLayouts    = (dnnLayout_t*)((*env)->GetPrimitiveArrayCritical(env, layouts, 0));
  dnnPrimitive_t primitive = NULL;
  dnnError_t status        = E_UNIMPLEMENTED;
  size_t jNumConcats       = (size_t)numConcats;

  status = dnnConcatCreate_F32(&primitive, NULL, jNumConcats, jLayouts);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, layouts, jLayouts, 0);

  return (long)primitive;
}
