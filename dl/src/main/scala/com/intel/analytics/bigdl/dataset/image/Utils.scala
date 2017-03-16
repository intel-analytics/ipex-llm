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

package com.intel.analytics.bigdl.dataset.image

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Font, Graphics2D}
import java.io.File
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.tensor.Tensor

/**
 * used for image object detection
 * visualize detected bounding boxes and their scores to image
 */
object Visualizer {
  private val bgColor = new Color(0, 0, 1, 127)
  private val font = new Font("Helvetica", Font.PLAIN, 14)
  private val stroke = new BasicStroke(3)

  private def vis(imgPath: String, className: String,
    scores: Tensor[Float], bboxes: Tensor[Float],
    savePath: String, thresh: Float = 0.3f): Unit = {
    var img: BufferedImage = null
    var g2d: Graphics2D = null

    def loadImage() = {
      img = ImageIO.read(new File(imgPath))
      g2d = img.createGraphics
      g2d.setFont(font)
      g2d.setStroke(stroke)
    }

    var i = 1
    while (i <= scores.size(1)) {
      val score = scores.valueAt(i)
      if (score > thresh) {
        if (g2d == null) {
          loadImage()
        }
        val bbox = bboxes(i)
        draw(g2d, bbox.valueAt(1).toInt, bbox.valueAt(2).toInt,
          bbox.valueAt(3).toInt - bbox.valueAt(1).toInt,
          bbox.valueAt(4).toInt - bbox.valueAt(2).toInt, s"$className ${ "%.3f".format(score) }")
      }
      i += 1
    }
    if (g2d != null) {
      ImageIO.write(img, savePath.substring(savePath.lastIndexOf(".") + 1), new File(savePath))
      g2d.dispose()
    }
  }

  private def draw(img: Graphics2D, x1: Int, y1: Int,
    width: Int, height: Int, title: String): Unit = {
    img.setColor(Color.RED)
    img.drawRect(x1, y1, width, height)

    val fm = img.getFontMetrics()
    val rect = fm.getStringBounds(title, img)

    img.setColor(bgColor)
    img.fillRect(x1, y1 - 2 - fm.getAscent,
      rect.getWidth.toInt,
      rect.getHeight.toInt)
    img.setColor(Color.WHITE)
    img.drawString(title, x1, y1 - 2)
  }


  /**
   * draw detected bounding boxes and scores of certain class to image
   * @param imagePath original image path
   * @param clsname   class name
   * @param scores    a series of detection scores
   * @param bboxes    a series of detection bounding boxes
   * @param thresh    thresh of scores to visualize
   * @param outPath   saved output image path
   */
  def visDetection(imagePath: String, clsname: String,
    scores: Tensor[Float], bboxes: Tensor[Float],
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    val f = new File(outPath)
    if (!f.exists()) {
      f.mkdirs()
    }
    val path = Paths.get(outPath,
      s"${ clsname }_${ imagePath.substring(imagePath.lastIndexOf("/") + 1) }").toString
    vis(imagePath, clsname, scores, bboxes, path, thresh)
  }
}
