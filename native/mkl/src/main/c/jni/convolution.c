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

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"
/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateForward
 * Signature: (III[I[I[I[I[II)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateForward
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));
  size_t *jOutputSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, outputSize, 0));
  size_t *jWeightSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, weightSize, 0));
  size_t *jStrides    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));
  jint *jPads       = (jint*)((*env)->GetPrimitiveArrayCritical(env, pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateForwardBias_F32(
    &primitive,
    NULL,
    dnnAlgorithmConvolutionDirect,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    dnnBorderZeros);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputSize, jOutputSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, weightSize, jWeightSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides   , jStrides   , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardData
 * Signature: (IJJ[J[J[J[J[II)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardData
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));
  size_t *jOutputSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, outputSize, 0));
  size_t *jWeightSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, weightSize, 0));
  size_t *jStrides    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));
  jint   *jPads       = (jint*)((*env)->GetPrimitiveArrayCritical(env, pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateBackwardData_F32(
    &primitive,
    NULL,
    jAlgorithm,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputSize, jOutputSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, weightSize, jWeightSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides   , jStrides   , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardKernel
 * Signature: (IJJ[J[J[J[J[II)J
 */
JNIEXPORT
jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardKernel
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray inputSize,
 jlongArray outputSize,
 jlongArray weightSize,
 jlongArray strides,
 jintArray pads,
 jint boderType)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));
  size_t *jOutputSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, outputSize, 0));
  size_t *jWeightSize = (size_t*)((*env)->GetPrimitiveArrayCritical(env, weightSize, 0));
  size_t *jStrides    = (size_t*)((*env)->GetPrimitiveArrayCritical(env, strides, 0));
  jint   *jPads       = (jint*)((*env)->GetPrimitiveArrayCritical(env, pads, 0));

  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  dnnBorder_t jBorderType   = (dnnBorder_t)boderType;
  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnGroupsConvolutionCreateBackwardFilter_F32(
    &primitive,
    NULL,
    jAlgorithm,
    groups,
    dim,
    jInputSize,
    jOutputSize,
    jWeightSize,
    jStrides,
    jPads,
    jBorderType);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, outputSize, jOutputSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, weightSize, jWeightSize, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, strides   , jStrides   , 0);
  (*env)->ReleasePrimitiveArrayCritical(env, pads      , jPads      , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    convolutionCreateBackwardBias
 * Signature: (IJJ[J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_convolutionCreateBackwardBias
(JNIEnv *env,
 jclass cls,
 jint algorithm,
 jlong groups,
 jlong dim,
 jlongArray outputSize)
{
  dnnAlgorithm_t jAlgorithm = (dnnAlgorithm_t)algorithm;
  size_t jGroups            = (size_t)groups;
  size_t jDim               = (size_t)dim;
  size_t *jOutputSize       = (size_t*)((*env)->GetPrimitiveArrayCritical(env, outputSize, 0));

  dnnError_t status        = E_UNIMPLEMENTED;
  dnnPrimitive_t primitive = NULL;

  status = dnnGroupsConvolutionCreateBackwardBias_F32(
    &primitive,
    NULL,
    jAlgorithm,
    jGroups,
    jDim,
    jOutputSize);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, outputSize, jOutputSize, 0);

  return (jlong)primitive;
}
