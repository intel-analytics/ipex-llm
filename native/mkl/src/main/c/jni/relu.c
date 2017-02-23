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
