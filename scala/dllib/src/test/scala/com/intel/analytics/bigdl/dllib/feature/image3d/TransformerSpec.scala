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
package com.intel.analytics.bigdl.dllib.feature.image3d

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.ImageFeature
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator._
import com.intel.analytics.bigdl.dllib.feature.common._
import com.intel.analytics.bigdl.dllib.feature.image._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.scalatest.{FlatSpec, Matchers}

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
    val dstTorch = Tensor[Double](Array[Double](0.0, 0.0, 0.29262194, 0.4249478, 0.31712994,
      0.09275303, 0.08961805, 0.0, 0.0, 0.0, 0.0, 0.29755187, 0.32578096, 0.5626346, 0.4365354,
      0.40343443, 0.41490048, 0.32817173, 0.22530137, 0.0, 0.0, 0.4672376, 0.4677945, 0.70909166,
      0.50209534, 0.6593656, 0.91172934, 0.7242027, 0.37856963, 0.39695457, 0.12731975, 0.66596985,
      0.70920634, 0.6457655, 0.35826272, 0.68768394, 0.5295299, 0.49672747, 0.5379895, 0.5760645,
      0.5341844, 0.4697795, 0.40747032, 0.438281, 0.2712102, 0.554312, 0.23995556, 0.36858487,
      0.40459204, 0.2524506, 0.4267454, 0.12342952, 0.2880426, 0.30744562, 0.17224228, 0.40548807,
      0.6973631, 0.27889606, 0.3665819, 0.63194805, 0.34321332, 0.08689031, 0.30138263, 0.4345351,
      0.81047827, 0.23823035, 0.62529624, 0.71889293, 0.56671214, 0.47234702, 0.73507214,
      0.40327334, 0.34665608, 0.39636195, 0.75153077, 0.51748544, 0.7237126, 0.90470463,
      0.87663966, 0.0, 0.0, 0.19828838, 0.34249613, 0.6051069, 0.66840446, 0.39873663,
      0.28909186, 0.5503567, 0.79466873, 0.0, 0.0, 0.0, 0.0, 0.43310156, 0.4053475, 0.5347002,
      0.674352, 0.37905228, 0.0, 0.0), Array(10, 10))
    val dstTensor = Tensor[Double](
      storage = Storage[Double](output(0)[Tensor[Float]](ImageFeature.imageTensor).storage().array()
        .map(_.toDouble)), storageOffset = 1, size = Array(1, 10, 10))
    dstTensor.view(10, 10).map(dstTorch, (v1, v2) => {
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
