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

class PvanetParam(phase: PhaseType = TEST) extends FasterRcnnParam(phase) {
  val anchorParam = AnchorParam(scales = Array[Float](3, 6, 9, 16, 32),
    ratios = Array(0.5f, 0.667f, 1.0f, 1.5f, 2.0f))

  override val SCALE_MULTIPLE_OF = 32
  SCALES = if (phase == TEST) Array(640)
  else Array(416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864)
  override val BBOX_VOTE = true
  override val NMS = 0.4f
  RPN_PRE_NMS_TOP_N = 12000
  RPN_POST_NMS_TOP_N = if (phase == TRAIN) 2000 else 200
  override val BG_THRESH_LO = 0.0
  BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true
  override val modelType: ModelType = Model.PVANET

}
