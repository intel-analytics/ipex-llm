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

package com.intel.analytics.bigdl.models.fasterrcnn.utils

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Font, Graphics2D}
import java.io.File
import java.nio.file.Paths
import javax.imageio.ImageIO

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.Target
import org.apache.log4j.Logger

object VisualizeTool {
  val logger = Logger.getLogger(getClass)

  private def vis(imgPath: String, clsname: String, dets: Target,
    savePath: String, thresh: Float = 0.3f): Unit = {
    var img: BufferedImage = null
    var g2d: Graphics2D = null

    def loadImage() = {
      img = ImageIO.read(new File(imgPath))
      g2d = img.createGraphics
      val font = new Font("Helvetica", Font.PLAIN, 14)
      g2d.setFont(font)
      g2d.setStroke(new BasicStroke(3))
    }

    var i = 1
    while (i <= Math.min(10, dets.classes.size(1))) {
      val bbox = dets.bboxes(i)
      val score = dets.classes.valueAt(i)
      if (score > thresh) {
        if (g2d == null) {
          loadImage()
        }
        //        logger.info(imgPath + s" $clsname with confidence: ${"%.3f".format(score)}")
        draw(g2d, bbox.valueAt(1).toInt, bbox.valueAt(2).toInt,
          bbox.valueAt(3).toInt - bbox.valueAt(1).toInt,
          bbox.valueAt(4).toInt - bbox.valueAt(2).toInt, s"$clsname ${ "%.3f".format(score) }")
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
    val bgColor = new Color(0, 0, 1, 127)

    val fm = img.getFontMetrics()
    val rect = fm.getStringBounds(title, img)

    img.setColor(bgColor)
    img.fillRect(x1, y1 - 2 - fm.getAscent,
      rect.getWidth.toInt,
      rect.getHeight.toInt)
    img.setColor(Color.WHITE)
    img.drawString(title, x1, y1 - 2)
  }


  def visDetection(imagePath: String, clsname: String, target: Target,
    thresh: Float = 0.3f, outPath: String = "data/demo"): Unit = {
    FileUtil.checkOrCreateDirs(outPath)
    vis(imagePath, clsname, target,
      Paths.get(outPath,
        s"${ clsname }_${ imagePath.substring(imagePath.lastIndexOf("/") + 1) }").toString, thresh)
  }
}
