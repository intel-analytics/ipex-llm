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

package com.intel.analytics.bigdl.models.fasterrcnn.tools

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import com.intel.analytics.bigdl.models.fasterrcnn.tools.Predictor._
import com.intel.analytics.bigdl.models.fasterrcnn.utils._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import org.apache.commons.lang3.SerializationUtils
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer

case class PredictorParam(NMS: Float, nClasses: Int,
  BBOX_VOTE: Boolean, maxPerImage: Int = 100, thresh: Double = 0.05)

object Predictor {
  val logger = Logger.getLogger(this.getClass)
}

class Predictor(param: PredictorParam) extends Serializable {

  @transient var nmsTool: Nms = _
  @transient var bboxTool: Bbox = _

  /**
   *
   * @param scores 21 * N
   * @param boxes  84 * N
   * @param param
   * @return
   */
  private def postProcess(scores: Tensor[Float], boxes: Tensor[Float], param: PredictorParam)
  : Array[Target] = {
    val results = new Array[Target](param.nClasses)
    // skip j = 0, because it's the background class
    var cls = 1
    while (cls < param.nClasses) {
      def getClsDet: Target = {
        val inds = (1 to scores.size(2)).filter(ind =>
          scores.valueAt(cls + 1, ind) > param.thresh).toArray
        if (inds.length == 0) return null
        val clsScores = TensorUtil.selectMatrix(scores(cls + 1), inds, 1)
        val clsBoxes = TensorUtil.selectMatrix(boxes.narrow(1, cls * 4 + 1, 4), inds, 2)

        val keepN = nmsTool.nms(clsScores, clsBoxes, param.NMS, inds)

        val bboxNms = TensorUtil.selectMatrix(clsBoxes.t(), inds, 1, keepN)
        val scoresNms = TensorUtil.selectMatrix(clsScores, inds, 1, keepN)
        if (param.BBOX_VOTE) {
          bboxTool.bboxVote(scoresNms, bboxNms, clsScores, clsBoxes.t())
        } else {
          Target(scoresNms, bboxNms)
        }
      }
      val clsDets = getClsDet
      results(cls) = clsDets
      cls += 1
    }

    // Limit to max_per_image detections *over all classes*
    if (param.maxPerImage > 0) {
      limitMaxPerImage(param, results)
    }
    results
  }

  def limitMaxPerImage(param: PredictorParam, results: Array[Target]): Unit = {
    val nImageScores = (1 until param.nClasses).map(j => if (results(j) == null) 0
    else results(j).classes.size(1)).sum
    if (nImageScores > param.maxPerImage) {
      val imageScores = ArrayBuffer[Float]()
      var j = 1
      while (j < param.nClasses) {
        val res = results(j).classes
        if (res.nElement() > 0) {
          res.apply1(x => {
            imageScores.append(x)
            x
          })
        }
        j += 1
      }
      val imageThresh = imageScores.sortWith(_ < _)(imageScores.length - param.maxPerImage)
      j = 1
      while (j < param.nClasses) {
        val box = results(j).bboxes
        val keep = (1 to box.size(1)).filter(x =>
          box.valueAt(x, box.size(2)) >= imageThresh).toArray
        val selectedScores = TensorUtil.selectMatrix(results(j).classes, keep, 1)
        val selectedBoxes = TensorUtil.selectMatrix(results(j).bboxes, keep, 1)
        results(j).classes.resizeAs(selectedScores).copy(selectedScores)
        results(j).bboxes.resizeAs(selectedBoxes).copy(selectedBoxes)
        j += 1
      }
    }
  }

  @transient var boxes: Tensor[Float] = _

  def imDetect(model: Module[Float], input: Table): Array[Target] = {
    var start = System.nanoTime()
    val result = model.forward(input).asInstanceOf[Table]
    logger.info(s"forward time is ${ (System.nanoTime() - start) / 1e9 }")
    start = System.nanoTime()

    val scores = result[Table](1)[Tensor[Float]](1)
    val boxDeltas = result[Table](1)[Tensor[Float]](2)
    val rois = result[Tensor[Float]](2)


    if (nmsTool == null) nmsTool = new Nms
    if (bboxTool == null) bboxTool = new Bbox
    // post process
    // unscale back to raw image space
    val imInfo = input[Tensor[Float]](2)
    if (boxes == null) boxes = Tensor[Float]
    boxes.resize(4, rois.size(1)).copy(rois.narrow(2, 2, 4).t()).div(imInfo.valueAt(3))
    // Apply bounding-box regression deltas
    val predBoxes = bboxTool.bboxTransformInv(boxes, boxDeltas.t())
    bboxTool.clipBoxes(predBoxes, imInfo.valueAt(1) / imInfo.valueAt(3),
      imInfo.valueAt(2) / imInfo.valueAt(4))
    val res = postProcess(scores.t(), predBoxes, param)
    logger.info(s"post process time is ${ (System.nanoTime() - start) / 1e9 } ")
    res
  }


  def clonePredictor(): Predictor = {
    SerializationUtils.clone(this)
  }
}
