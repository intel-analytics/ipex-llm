/*
 * Copyright 2018 Analytics Zoo Authors.
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
package com.intel.analytics.zoo.feature.image3d

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFeature}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.TH
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.zoo.feature.common._
import com.intel.analytics.zoo.feature.image._
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Dense, Flatten, Input}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.nnframes.{NNClassifierModel}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SQLContext}
import org.opencv.core.CvType

import scala.reflect.ClassTag

class TransformerSpec extends FlatSpec with Matchers{
  "A pipleline transformer with ImageSet" should "generate correct output." in{
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Float](10, 20, 30)
    input.apply1(e => RNG.uniform(0, 1).toFloat)
    input.resize(10, 20, 30, 1)
    val image = ImageFeature3D(input)
    val imageClone = image.clone()
    val imageSet = ImageSet.array(Array[ImageFeature3D](image).map(_.asInstanceOf[ImageFeature]))
    // chain 3d transformer
    val start = Array[Int](5, 10, 10)
    val patchSize = Array[Int](1, 10, 10)
    val rotAngles = Array[Double](0, 0, math.Pi/3.7)
    val chainTransformer = Crop3D(start, patchSize) -> Rotate3D(rotAngles)
    val output = chainTransformer(imageSet).toLocal().array


    // get cropped image for torch rotation
    val cropper = Crop3D(start, patchSize)
    val cropped_image = cropper.transform(imageClone)

    // compare chain transformer result with torch result
    val code = "require 'image'\n" +
      "dst = image.rotate(src,math.pi/3.7,'bilinear')"
    val (luaTime, torchResult) = TH.run(code,
      Map("src" -> cropped_image[Tensor[Float]](ImageFeature.imageTensor).clone.view(10, 10)),
      Array("dst"))
    val dstTorch = torchResult("dst").asInstanceOf[Tensor[Float]]
    output(0).asInstanceOf[ImageFeature3D][Tensor[Float]](ImageFeature.imageTensor)
      .view(10, 10).map(dstTorch, (v1, v2) => {
      assert(math.abs(v1-v2)<1e-6)
      v1
    })
  }

//  "A pipleline transformer with dataframe" should "generate correct output." in{
//    val seed = 100
//    RNG.setSeed(seed)
//    val src = Tensor[Float](10, 20, 30, 1)
//    src.apply1(e => RNG.uniform(0, 1).toFloat)
//    val sc = NNContext.initNNContext("test")
//    val image = ImageFeature3D(src)
//    val imageRDD = sc.parallelize(Seq(image))
//    val imageSet = ImageSet.rdd(imageRDD.map(_.asInstanceOf[ImageFeature]))
//    val rowRDD = imageSet.toDistributed().rdd.map { imf =>
//      Row(imf3D2Row(imf.asInstanceOf[ImageFeature3D]))
//    }
//    val imageDF = SQLContext.getOrCreate(sc).createDataFrame(rowRDD, imageColumnSchema)
//
//    // chain 3d transformer
//    val start = Array[Int](5, 10, 10)
//    val patchSize = Array[Int](1, 10, 10)
//    val rotAngles = Array[Double](0, 0, math.Pi/3.7)
//    val transformer = RowToImageFeature3D() -> Crop3D(start, patchSize) ->
//      Rotate3D(rotAngles) -> ImageFeatureToTensor()
//
//    // create model
//    val input = Input[Float](inputShape = Shape(1, 10, 10, 1))
//    val flatten = Flatten[Float]().inputs(input)
//    val logits = Dense[Float](2).inputs(flatten)
//    val model = Model[Float](input, logits)
//
//    val dlmodel = NNClassifierModel(model = model, featurePreprocessing = transformer)
//      .setBatchSize(1)
//      .setFeaturesCol("image")
//      .setPredictionCol("prediction")
//
//    val resultDF = dlmodel.transform(imageDF)
//    resultDF.select("image", "prediction").show(10, false)
//
//  }

  private def imf3D2Row(imf: ImageFeature3D): Row = {
    val (mode, data) = if (imf.contains(ImageFeature.imageTensor)) {
      val floatData = imf(ImageFeature.imageTensor).asInstanceOf[Tensor[Float]].storage().array()
      val cvType = imf.getChannel()
      (cvType, floatData)
    } else if (imf.contains(ImageFeature.bytes)) {
      val bytesData = imf.bytes()
      val cvType = imf.getChannel()
      (cvType, bytesData)
    } else {
      throw new IllegalArgumentException(s"ImageFeature should have imageTensor or bytes.")
    }

    Row(
      imf.uri(),
      imf.getDepth(),
      imf.getHeight(),
      imf.getWidth(),
      imf.getChannel(),
      mode,
      data
    )
  }

  private val imageColumnSchema =
    StructType(StructField("image", floatSchema, true) :: Nil)

  private val floatSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("depth", IntegerType, false) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // floats in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", new ArrayType(FloatType, false), false) :: Nil)

  private def row2IMF(row: Row): ImageFeature = {
    val (origin, d, h, w, c) = (row.getString(0), row.getInt(1), row.getInt(2),
      row.getInt(3), row.getInt(4))
    val imf = ImageFeature3D()
    imf.update(ImageFeature.uri, origin)
    imf.update(ImageFeature.size, Array(d, h, w, c))
    val data = row.getSeq[Float](6).toArray
    val size = Array(d, h, w, c)
    val ten = Tensor(Storage(data)).resize(size)
    imf.update(ImageFeature.imageTensor, ten)
    imf
  }

  class RowToImageFeature3D()
    extends Preprocessing[Row, ImageFeature] {

    override def apply(prev: Iterator[Row]): Iterator[ImageFeature] = {
      prev.map { row =>
        row2IMF(row)
      }
    }
  }

  object RowToImageFeature3D {
    def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): RowToImageFeature3D =
      new RowToImageFeature3D()
  }
}
