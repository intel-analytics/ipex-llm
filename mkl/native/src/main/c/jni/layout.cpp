#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutCreate
 * Signature: (I[I[I)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutCreate
  (JNIEnv *env, jclass cls, jint dimension, jlongArray size, jlongArray strides)
{
  size_t *jSize    = (size_t*)(env->GetPrimitiveArrayCritical(size, 0));
  size_t *jStrides = (size_t*)(env->GetPrimitiveArrayCritical(strides, 0));

  for (int i = 0; i < dimension; i++) {
    LOG(INFO) << "size[" << i << "] = " << jSize[i];
  }

  for (int i = 0; i < dimension; i++) {
    LOG(INFO) << "strides[" << i << "] = " << jStrides[i];
  }

  dnnError_t status  = E_UNIMPLEMENTED;
  dnnLayout_t layout = NULL;

  status = dnnLayoutCreate_F32(&layout, dimension, jSize, jStrides);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(size, jSize, 0);
  env->ReleasePrimitiveArrayCritical(strides, jStrides, 0);
  
  return (jlong)layout;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutCreateFromPrimitive
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutCreateFromPrimitive
  (JNIEnv *env, jclass cls, jlong primitive, jint type)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)(primitive);
  dnnLayout_t layout        = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;
  dnnResourceType_t jType   = (dnnResourceType_t)type;

  status = dnnLayoutCreateFromPrimitive_F32(&layout, jPrimitive, jType);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)layout;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutGetMemorySize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutGetMemorySize
  (JNIEnv *env, jclass cls, jlong layout)
{
  dnnLayout_t jLayout = (dnnLayout_t)(layout);

  size_t size = dnnLayoutGetMemorySize_F32(jLayout);

  return (long)size;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutCompare
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutCompare
  (JNIEnv *env, jclass cls, jlong layout1, jlong layout2)
{
  dnnLayout_t jLayout1 = (dnnLayout_t)layout1;
  dnnLayout_t jLayout2 = (dnnLayout_t)layout2;

  return (int)dnnLayoutCompare_F32(jLayout1, jLayout2);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionCreate
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionCreate
  (JNIEnv *env, jclass cls, jlong layout1, jlong layout2)
{
  dnnLayout_t jLayout1 = (dnnLayout_t)layout1;
  dnnLayout_t jLayout2 = (dnnLayout_t)layout2;

  LOG(INFO) << jLayout1;
  LOG(INFO) << jLayout2;

  dnnPrimitive_t primitive = NULL;
  dnnError_t status        = E_UNIMPLEMENTED;

  status = dnnConversionCreate_F32(&primitive, jLayout1, jLayout2);
  CHECK_EQ(status, E_SUCCESS);

  LOG(INFO) << status;
  LOG(INFO) << primitive;

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionExecuteToMkl
 * Signature: ([F[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionExecuteToMkl
  (JNIEnv *env, jclass cls, jfloatArray usr, jint offset, jfloatArray mkl, jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jUsr = (jfloat*)(env->GetPrimitiveArrayCritical(usr, 0));
  jfloat* jMkl = (jfloat*)(env->GetPrimitiveArrayCritical(mkl, 0));

  LOG(INFO) << jMkl;
  LOG(INFO) << jUsr;

  void *resources[dnnResourceNumber];

  resources[dnnResourceFrom] = (void*)(jUsr + offset);
  resources[dnnResourceTo]   = (void*)jMkl;

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(usr, jUsr, 0);
  env->ReleasePrimitiveArrayCritical(mkl, jMkl, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionExecuteToUsr
 * Signature: ([F[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionExecuteToUsr
  (JNIEnv *env, jclass cls, jfloatArray usr, jint offset, jfloatArray mkl, jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jUsr = (jfloat*)(env->GetPrimitiveArrayCritical(usr, 0));
  jfloat* jMkl = (jfloat*)(env->GetPrimitiveArrayCritical(mkl, 0));

  void *resources[dnnResourceNumber];

  resources[dnnResourceFrom] = (void*)jMkl;
  resources[dnnResourceTo]   = (void*)(jUsr + offset);

  status = dnnExecute_F32(jPrimitive, resources);
  CHECK_EQ(status, E_SUCCESS);

  env->ReleasePrimitiveArrayCritical(usr, jUsr, 0);
  env->ReleasePrimitiveArrayCritical(mkl, jMkl, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    deletePrimitive
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_deletePrimitive
  (JNIEnv *env, jclass cls, jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status = E_UNIMPLEMENTED;
  
  status = dnnDelete_F32(jPrimitive);
  CHECK_EQ(status, E_SUCCESS);
}

