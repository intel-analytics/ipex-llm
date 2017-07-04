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
package com.intel.analytics.bigdl.integration

import com.intel.analytics.bigdl.models.lenet.LeNet5
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.nn.{Linear, Module, Sequential}
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble
import org.apache.commons.compress.utils.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Integration
class S3Spec extends FlatSpec with Matchers with BeforeAndAfter{
  val s3aPath = System.getProperty("s3aPath")

  private def processPath(path: String): String = {
    if (path.contains(":")) {
      path.substring(1)
    } else {
      path
    }
  }

  "save and load model from s3" should "be correct" in {
    val path = s3aPath + "/lenet.obj"
    val model = LeNet5(10)
    model.save(path, true)
    val lenet = Module.load(path)
    lenet should be (model)
  }

  "load caffe model from s3" should "work properly" in {
    val prototxt = getClass().getClassLoader().getResource("caffe/test.prototxt").getPath
    val modelPath = getClass().getClassLoader().getResource("caffe/test.caffemodel").getPath

    val hdfsDir = s3aPath + s"/${ com.google.common.io.Files.createTempDir().getPath() }"

    def writeToHdfs(localFile: String, hdfsDir: String): Unit = {
      val src = new Path(localFile)
      val fs = src.getFileSystem(new Configuration(false))
      val inStream = fs.open(src)
      val dest = new Path(hdfsDir)
      val fsDest = dest.getFileSystem(new Configuration())
      val outFileStream = fsDest.create(dest)

      IOUtils.copy(inStream, outFileStream)

      // Close both files
      inStream.close()
      outFileStream.close()
    }

    writeToHdfs(prototxt, hdfsDir + "/test.prototxt")
    writeToHdfs(modelPath, hdfsDir + "/test.caffemodel")
    val module = Sequential()
      .add(Convolution(3, 4, 2, 2).setName("conv"))
      .add(Convolution(4, 3, 2, 2).setName("conv2"))
      .add(Linear(2, 27, withBias = false).setName("ip"))


    val model = CaffeLoader.load[Double](module, prototxt, modelPath)

    val modelFromHdfs = CaffeLoader.load[Double](module, hdfsDir + "/test.prototxt",
      hdfsDir + "/test.caffemodel")

    model.getParameters() should be (modelFromHdfs.getParameters())

  }
}
