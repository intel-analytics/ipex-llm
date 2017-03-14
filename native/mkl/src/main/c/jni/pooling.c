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
#include <string.h>

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

#include <omp.h>
#include <sched.h>

static void unPadding(float* from, float* to, size_t *fromStrides,
                      size_t *toSize, size_t *toStrides)
{
  int n, c, h;
  for (n = 0; n < toSize[3]; n++)
    for (c = 0; c < toSize[2]; c++) {
      int baseToIndex = n * toStrides[3] + c * toStrides[2];
      int baseFromIndex = n * fromStrides[3] + c * fromStrides[2];

#pragma omp parallel for
      for (h = 0; h < toSize[1]; h++) {
        memcpy(to + baseToIndex + h * toStrides[1],
               from + baseFromIndex + h * fromStrides[1],
               toSize[0] * sizeof(float));
      }
    }
}

static void padding(float* from, float* to, size_t *fromSize,
                    size_t *fromStrides, size_t *toSize, size_t *toStrides)
{
  int n, c, h;
  for (n = 0; n < fromSize[3]; n++) {
    for (c = 0; c < fromSize[2]; c++) {
      int baseToIndex = n * toStrides[3] + c * toStrides[2];
      int baseFromIndex = n * fromStrides[3] + c * fromStrides[2];

#pragma omp parallel for
      for (h = 0; h < fromSize[1]; h++) {  // height
        memcpy(to + baseToIndex + h * toStrides[1],
               from + baseFromIndex + h * fromStrides[1],
               fromSize[0] * sizeof(float));

        // the last column of a matrix with 0. we only need to set
        // one element to 0, because 0 <= ceil - floor <= 1
        if (toSize[0] > fromSize[0]) {
          int end     = baseToIndex + h * toStrides[1] + fromSize[0];
          *(to + end) = 0;
        }
      }

      // pad the last row of a matrix with 0 * width
      if (toSize[1] > fromSize[1]) {
        int end = baseToIndex + fromSize[1] * toStrides[1];
        memset(to + end, 0, toSize[0] * sizeof(float));
      }
    }
  }
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    unPadding
 * Signature: ([FJJ[J[J[J)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_unPadding
(JNIEnv *env,
 jclass cls,
 jfloatArray to,
 jlong offset,
 jlong from,
 jlongArray fromStrides,
 jlongArray toSize,
 jlongArray toStrides)
{
  jfloat* jTo = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, to, 0));
  size_t * jFromStrides = (size_t*)((*env)->GetPrimitiveArrayCritical(env,
                                                                   fromStrides,
                                                                   0));
  size_t* jToSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, toSize, 0));
  size_t* jToStrides = (size_t*)((*env)->GetPrimitiveArrayCritical(env, toStrides,
                                                                 0));

  jfloat *jFrom = (jfloat*)(from);
  
  unPadding(jFrom, jTo + offset, jFromStrides, jToSize, jToStrides);

  (*env)->ReleasePrimitiveArrayCritical(env, to, jTo, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, fromStrides, jFromStrides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, toSize, jToSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, toStrides, jToStrides, 0);
}


/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    padding
 * Signature: ([FJJ[J[J[J[J)V
 */
JNIEXPORT
  void JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_padding
(JNIEnv *env,
 jclass cls,
 jfloatArray from,
 jlong offset,
 jlong to,
 jlongArray fromSize,
 jlongArray fromStrides,
 jlongArray toSize,
 jlongArray toStrides)
{
  jfloat *jFrom = (jfloat *)((*env)->GetPrimitiveArrayCritical(env, from, 0));
  size_t* jFromSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, fromSize,
                                                                0));
  size_t* jFromStrides = (size_t*)((*env)->GetPrimitiveArrayCritical(env,
                                                                   fromStrides,
                                                                   0));
  size_t* jToSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, toSize, 0));
  size_t* jToStrides = (size_t*)((*env)->GetPrimitiveArrayCritical(env, toStrides,
                                                                 0));
  jfloat *jTo = (jfloat*)(to);

  padding(jFrom + offset, jTo, jFromSize, jFromStrides, jToSize, jToStrides);

  (*env)->ReleasePrimitiveArrayCritical(env, from, jFrom, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, fromSize, jFromSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, fromStrides, jFromStrides, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, toSize, jToSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, toStrides, jToStrides, 0);
}
