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
import com.intel.analytics.bigdl.models.fasterrcnn.model.Phase.PhaseType
import com.intel.analytics.bigdl.models.fasterrcnn.utils.AnchorParam
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}


object Phase extends Enumeration {
  type PhaseType = Value
  val TRAIN, TEST = Value

}

object Model extends Enumeration {
  type ModelType = Value
  val VGG16, PVANET = Value
}

abstract class FasterRcnnParam(val phase: PhaseType = Phase.TEST) {
  val anchorParam: AnchorParam

  // Scales to use during training (can list multiple scales)
  // Each scale is the pixel size of an image"s shortest side
  var SCALES = Array(600)

  // Resize test images so that its width and height are multiples of ...
  val SCALE_MULTIPLE_OF = 1

  // Minibatch size (number of regions of interest [ROIs])
  val BATCH_SIZE = 3

  // Fraction of minibatch that is labeled foreground (i.e. class > 0)
  val FG_FRACTION = 0.25

  // Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
  val FG_THRESH = 0.5f

  // Overlap threshold for a ROI to be considered background (class = 0 if
  // overlap in [LO, HI))
  val BG_THRESH_HI = 0.5
  val BG_THRESH_LO = 0.1


  // Normalize the targets using "precomputed" (or made up) means and stdevs
  // (BBOX_NORMALIZE_TARGETS must also be true)
  var BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true

  // If an anchor statisfied by positive and negative conditions set to negative
//  val RPN_CLOBBER_POSITIVES = false
  // Number of top scoring boxes to keep before apply NMS to RPN proposals
  var RPN_PRE_NMS_TOP_N = 12000
  // Number of top scoring boxes to keep after applying NMS to RPN proposals
  var RPN_POST_NMS_TOP_N = 2000

  // Overlap threshold used for non-maximum suppression (suppress boxes with
  // IoU >= this threshold)
  val NMS = 0.3f

  // Apply bounding box voting
  val BBOX_VOTE = false

  val modelType: ModelType
}




