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
