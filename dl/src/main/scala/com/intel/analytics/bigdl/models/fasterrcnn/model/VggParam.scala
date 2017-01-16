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

package com.intel.analytics.bigdl.models.fasterrcnn.model

import com.intel.analytics.bigdl.models.fasterrcnn.model.Model.ModelType
import com.intel.analytics.bigdl.models.fasterrcnn.model.Phase._
import com.intel.analytics.bigdl.models.fasterrcnn.utils.AnchorParam

class VggParam(phase: PhaseType = TEST) extends FasterRcnnParam(phase) {
  val anchorParam = AnchorParam(scales = Array[Float](8, 16, 32),
    ratios = Array[Float](0.5f, 1.0f, 2.0f))
  override val BG_THRESH_LO = if (phase == TRAIN) 0.0 else 0.1
  override val BATCH_SIZE = 128
  override val modelType: ModelType = Model.VGG16

  RPN_PRE_NMS_TOP_N = if (phase == TRAIN) 12000 else 6000
  RPN_POST_NMS_TOP_N = if (phase == TRAIN) 2000 else 300
}


