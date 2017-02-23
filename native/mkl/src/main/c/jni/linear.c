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

#include "com_intel_analytics_bigdl_mkl_MklDnnFloat.h"
#include "debug.h"

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateForwardBias
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateForwardBias
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateForwardBias_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackData
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackData
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardData_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackWeight
 * Signature: (J[JJ)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackWeight
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray inputSize,
 jlong outputChannel)
{
  size_t *jInputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, inputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardFilter_F32(
    &primitive,
    NULL,
    dimension,
    jInputSize,
    outputChannel);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, inputSize , jInputSize , 0);

  return (jlong)primitive;
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MklDnnFloat
 * Method:    linearCreateBackBias
 * Signature: (J[J)J
 */
JNIEXPORT
  jlong JNICALL Java_com_intel_analytics_bigdl_mkl_MklDnnFloat_linearCreateBackBias
(JNIEnv *env,
 jclass cls,
 jlong dimension,
 jlongArray outputSize)
{
  size_t *jOutputSize  = (size_t*)((*env)->GetPrimitiveArrayCritical(env, outputSize, 0));

  dnnPrimitive_t primitive  = NULL;
  dnnError_t status         = E_UNIMPLEMENTED;

  status = dnnInnerProductCreateBackwardBias_F32(
    &primitive,
    NULL,
    dimension,
    jOutputSize);
  CHECK_EQ(status, E_SUCCESS);

  (*env)->ReleasePrimitiveArrayCritical(env, outputSize , jOutputSize , 0);

  return (jlong)primitive;
}
