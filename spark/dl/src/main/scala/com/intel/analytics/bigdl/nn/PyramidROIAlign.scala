/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

@SerialVersionUID(-1562995431845030993L)
class PyramidROIAlign(val poolH: Int, val poolW: Int, val imgH: Int, val imgW: Int, val imgC: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {
  val concat = JoinTable(1, 4)
  val concat2 = JoinTable[Int](1, 2)

  override def updateOutput(input: Table): Tensor[Float] = {
    // Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    val boxesInput = input[Tensor[Float]](1)
    val boxes = boxesInput.view(boxesInput.size()).squeeze(1)
    val channelDim = 2
    // Assign each ROI to a level in the pyramid based on the ROI area
    val splits = boxes.split(2)
    val x1 = splits(0)
    val y1 = splits(1)
    val x2 = splits(2)
    val y2 = splits(3)
    // todo: optimize
    val h = y2 - y1
    val w = x2 - x1
    // Equation 1 in the Feature Pyramid Networks paper. Account for
    // the fact that our coordinates are normalized here.
    // e.g. a 224x224 ROI (in pixels) maps to P4
    val imageArea = imgH * imgW
    val roiLevel = log2((h.cmul(w)).sqrt() / (224.0f / Math.sqrt(imageArea).toFloat))
    roiLevel.apply1(x => {
      if (x.equals(Float.NaN)) Float.NegativeInfinity else Math.round(x)
    })
    roiLevel.apply1(x => {
      Math.min(5, Math.max(2, 4 + x.round))
    })
    roiLevel.squeeze()
    // Loop through levels and apply ROI pooling to each. P2 to P5.
    var i = 1
    var level = 2
    val boxToLevel = T()
    val pooledTable = T()
    while (i <= input.length() - 1) {
      val ix = (1 to roiLevel.nElement()).filter(roiLevel.valueAt(_) == level).toArray
      if (ix.length > 0) {
        boxToLevel.insert(Tensor[Int](Storage(ix)).resize(ix.length, 1))
      }

      val cropResize = Tensor[Float](ix.length, input[Tensor[Float]](2).size(channelDim),
        poolH, poolW)
      val featureMap = input[Tensor[Float]](i + 1)
      ix.zip(Stream from (1)).foreach(ind => {
        val box = boxes(ind._1)
        PyramidROIAlign.cropAndResize(featureMap, box, poolH, poolW, cropResize(ind._2))
      })
      if (cropResize.nElement() > 0) pooledTable.insert(cropResize)

      i += 1
      level += 1
    }
    // Pack pooled features into one tensor
    val pooled = concat.forward(pooledTable).asInstanceOf[Tensor[Float]]
    val boxToLevels = concat2.forward(boxToLevel)
    val ix = boxToLevels.squeeze()
      .toArray().asInstanceOf[Array[Int]].zipWithIndex.sortBy(_._1).map(_._2)

    i = 0
    output.resizeAs(pooled)
    while (i < ix.length) {
      i += 1
      output(i).copy(pooled(ix(i - 1) + 1))
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new NotImplementedError()
  }

  private def log2(x: Tensor[Float]): Tensor[Float] = {
    x.log().div(Math.log(2).toFloat)
  }
}

object PyramidROIAlign {
  def apply(poolH: Int, poolW: Int, imgH: Int, imgW: Int, imgC: Int)
    (implicit ev: TensorNumeric[Float]): PyramidROIAlign =
    new PyramidROIAlign(poolH, poolW, imgH, imgW, imgC)

  def cropAndResize(image: Tensor[Float],
    box: Tensor[Float], cropH: Int, cropW: Int, crops: Tensor[Float])
    (implicit ev: TensorNumeric[Float]): Unit = {
    val imageH = image.size(3)
    val imageW = image.size(4)
    val depth = image.size(2)
    val x1 = box.valueAt(1)
    val y1 = box.valueAt(2)
    val x2 = box.valueAt(3)
    val y2 = box.valueAt(4)
    val b_in = 1

    val heightScale = if (cropH > 1) {
      (y2 - y1) * (imageH - 1) / (cropH - 1)
    } else {
      0
    }
    val widthScale = if (cropW > 1) {
      (x2 - x1) * (imageW - 1) / (cropW - 1)
    } else {
      0
    }
    var y = 0
    while (y < cropH) {
      val inY = if (cropH > 1) {
        y1 * (imageH - 1) + y * heightScale
      } else {
        0.5 * (y1 + y2) * (imageH - 1)
      }
      if (!(inY < 0 || inY > imageH - 1)) {
        val topYIndex = inY.floor.toInt
        val bottomYIndex = inY.ceil.toInt
        val yLerp = inY - topYIndex
        var x = 0
        while (x < cropW) {
          val inX = if (cropW > 1) {
            x1 * (imageW - 1) + x * widthScale
          } else {
            0.5 * (x1 + x2) * (imageW - 1)
          }
          if (!(inX < 0 || inX > imageW - 1)) {
            val leftXIndex = inX.floor.toInt
            val rightXIndex = inX.ceil.toInt
            val xLerp = inX - leftXIndex
            var d = 0
            while (d < depth) {
              val topLeft = image.valueAt(b_in, d + 1, topYIndex + 1, leftXIndex + 1)
              val topRight = image.valueAt(b_in, d + 1, topYIndex + 1, rightXIndex + 1)
              val bottomLeft = image.valueAt(b_in, d + 1, bottomYIndex + 1, leftXIndex + 1)
              val bottomRight = image.valueAt(b_in, d + 1, bottomYIndex + 1, rightXIndex + 1)
              val top = topLeft + (topRight - topLeft) * xLerp
              val bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp
              crops.setValue(d + 1, y + 1, x + 1, (top + (bottom - top) * yLerp).toFloat)
              d += 1
            }
          }
          x += 1
        }
      }
      y += 1
    }
  }
}
