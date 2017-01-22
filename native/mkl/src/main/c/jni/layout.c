/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mkl_dnn.h>
#include <mkl_dnn_types.h>
#include <mkl_service.h>

#include <omp.h>
#include <sched.h>

#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */
#include <time.h>

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

#ifdef PERF
const int INPERF = 1;
#else
const int INPERF = 0;
#endif

#define PERFSTART() \
  do { \
    struct timespec start, end; \
    if (INPERF) { \
      clock_gettime(CLOCK_MONOTONIC, &start); \
    }

#define PERFEND(x) \
    if (INPERF) { \
      clock_gettime(CLOCK_MONOTONIC, &end); \
      fprintf(stderr, x " %lf\n", (end.tv_sec - start.tv_sec) * 1000 + \
              (double)(end.tv_nsec - start.tv_nsec) / 1000000); \
    } \
  } while(0);

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutCreate
 * Signature: (I[J[J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutCreate
  (JNIEnv *env, jclass cls, jint dimension, jlongArray size, jlongArray strides)
{
  size_t *jSize    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, size, 0));
  size_t *jStrides = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));

  dnnError_t status  = E_UNIMPLEMENTED;
  dnnLayout_t layout = NULL;

  status = dnnLayoutCreate_F32(&layout, dimension, jSize, jStrides);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, size, jSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides, jStrides, 0);
  
  return (jlong)layout;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    layoutDelete
 * Signature: (J)J
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_layoutDelete
  (JNIEnv *env, jclass cls, jlong layout)
{
  dnnLayout_t jLayout = (dnnLayout_t)layout;
  dnnLayoutDelete_F32(jLayout);
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

  dnnPrimitive_t primitive = NULL;
  dnnError_t status        = E_UNIMPLEMENTED;

  status = dnnConversionCreate_F32(&primitive, jLayout1, jLayout2);
  CHECK_EQ(status, E_SUCCESS);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionExecuteToMkl
 * Signature: ([F[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionExecuteToMkl
  (JNIEnv *env, jclass cls, jfloatArray usr, jint offset, jlong mkl, jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jUsr = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, usr, 0));
  jfloat* jMkl = (jfloat*)(mkl);

  void *resources[dnnResourceNumber];

  resources[dnnResourceFrom] = (void*)(jUsr + offset);
  resources[dnnResourceTo]   = (void*)jMkl;

  PERFSTART();
  status = dnnExecute_F32(jPrimitive, resources);
  PERFEND("conversion usr->mkl costs");
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, usr, jUsr, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionExecuteToUsr
 * Signature: ([F[FJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionExecuteToUsr
  (JNIEnv *env, jclass cls, jfloatArray usr, jint offset, jlong mkl, jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  jfloat* jUsr = (jfloat*)((*env)->GetPrimitiveArrayCritical(env, usr, 0));
  jfloat* jMkl = (jfloat*)(mkl);

  void *resources[dnnResourceNumber];

  resources[dnnResourceFrom] = (void*)jMkl;
  resources[dnnResourceTo]   = (void*)(jUsr + offset);

  PERFSTART();
  status = dnnExecute_F32(jPrimitive, resources);
  PERFEND("conversion mkl->usr costs");
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, usr, jUsr, 0);
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

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    setZero
 * Signature: ([F)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_setZero
  (JNIEnv *env,
   jclass cls,
   jlong buffer,
   jlong len)
{
  jfloat* jBuffer = (jfloat*)buffer;
  jlong jLen = len / sizeof(jfloat);
#pragma omp parallel for
  for (int i = 0; i < jLen; ++i) {
    jBuffer[i] = 0;
  }
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    conversionExecuteMklToMkl
 * Signature: (JJJ)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_conversionExecuteMklToMkl
(JNIEnv *env,
 jclass cls,
 jlong src,
 jlong dst,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;
  dnnError_t status         = E_UNIMPLEMENTED;

  void* jSrc = (void*)src;
  void* jDst = (void*)dst;

  void *resources[dnnResourceNumber];

  resources[dnnResourceFrom] = (void*)jSrc;
  resources[dnnResourceTo]   = (void*)jDst;

  PERFSTART();
  status = dnnExecute_F32(jPrimitive, resources);
  PERFEND("conversion mkl->mkl costs");
  CHECK_EQ(status, E_SUCCESS);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    execute
 * Signature: ([JJ)I
 */
JNIEXPORT
jint JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_execute
(JNIEnv *env,
 jclass cls,
 jlongArray resources,
 jlong primitive)
{
  dnnPrimitive_t jPrimitive = (dnnPrimitive_t)primitive;

  jlong* jResources = (jlong*)((*env)->GetPrimitiveArrayCritical(env, resources, 0));

  // all pointer to void*
  void* res[dnnResourceNumber];
  for(int i = 0; i < dnnResourceNumber; i++){
    res[i] = (void*)jResources[i];
  }
  dnnError_t status = E_UNIMPLEMENTED;
  PERFSTART();
  status = dnnExecute_F32(jPrimitive, res);
  PERFEND("execute costs");
  CHECK_EQ(status, E_SUCCESS);
  (*env)->ReleasePrimitiveArrayCritical(env, resources, jResources, 0);

  return status;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    allocateBuffer
 * Signature: (J)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_allocateBuffer
(JNIEnv *env,
 jclass cls,
 jlong layout)
{
  dnnLayout_t jLayout = (dnnLayout_t)layout;
  dnnError_t status = E_UNIMPLEMENTED;
  void *data = NULL;
  status = dnnAllocateBuffer_F32(&data, jLayout);
  CHECK_EQ(status, E_SUCCESS);
  
  return (jlong)data;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    releaseBuffer
 * Signature: (J)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_releaseBuffer
(JNIEnv *env,
 jclass cls,
 jlong data)
{
  if (data) {
    void* jData = (void*)data;
    dnnError_t status = E_UNIMPLEMENTED;
    status = dnnReleaseBuffer_F32(jData);
    CHECK_EQ(status, E_SUCCESS);
  }
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    buffercpy
 * Signature: (JJJ)V
 */
JNIEXPORT
void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_buffercpy
(JNIEnv *env,
 jclass cls,
 jlong dst,
 jlong src,
 jlong len)
{
  // memcpy((void*)dst, (void*)src, sizeof(jfloat) * len);
  jfloat* jDst = (jfloat*)dst;
  jfloat* jSrc = (jfloat*)src;
  jlong jLen = (jlong)len / sizeof(jfloat);

#pragma omp parallel for
  for (int i = 0; i < jLen; ++i) {
    jDst[i] = jSrc[i];
  }
}
