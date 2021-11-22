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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential._
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest
import org.apache.spark.SparkConf
import org.apache.spark.serializer.KryoSerializer
import org.scalatest.{FlatSpec, Matchers}

class WordEmbeddingSpec extends FlatSpec with Matchers {
  val gloveDir: String = getClass.getClassLoader.getResource("glove.6B").getPath
  val embeddingFile: String = gloveDir + "/glove.6B.50d.txt"

  "WordEmbedding GloVe with wordIndex and serialization" should "work properly" in {
    val wordIndex = Map("the" -> 1, "with" -> 2, "analyticszoo" -> 3)
    val seq = Sequential[Float]()
    val layer = WordEmbedding[Float](embeddingFile, wordIndex, inputLength = 1)
    seq.add(layer)
    val input = Tensor[Float](4, 1)
    input(Array(1, 1)) = 1
    input(Array(2, 1)) = 3 // not in GloVe
    input(Array(3, 1)) = 2
    input(Array(4, 1)) = 0 // unknown word
    val output = seq.forward(input).toTensor[Float]
    output.size() should be (Array(4, 1, 50))
    val expected = Array(0.418, 0.24968, -0.41242, 0.1217, 0.34527,
      -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
      0.27843, -0.14767, -0.55677, 0.14658, -0.0095095,
      0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
      -0.016801, -0.33279, -0.1552, -0.23131, -0.19181,
      -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
      4.0071, -0.18594, -0.52287, -0.31681, 0.00059213,
      0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
      -0.29871, -0.15749, -0.34758, -0.045637, -0.44251,
      0.18785, 0.0027849, -0.18411, -0.11514, -0.78581,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.25616, 0.43694, -0.11889, 0.20345, 0.41959,
      0.85863, -0.60344, -0.31835, -0.6718, 0.003984,
      -0.075159, 0.11043, -0.73534, 0.27436, 0.054015,
      -0.23828, -0.13767, 0.011573, -0.46623, -0.55233,
      0.083317, 0.55938, 0.51903, -0.27065, -0.28211,
      -1.3918, 0.17498, 0.26586, 0.061449, -0.273,
      3.9032, 0.38169, -0.056009, -0.004425, 0.24033,
      0.30675, -0.12638, 0.33436, 0.075485, -0.036218,
      0.13691, 0.37762, -0.12159, -0.13808, 0.19505,
      0.22793, -0.17304, -0.07573, -0.25868, -0.39339,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0)
    output.storage().toArray should be (expected.map(_.toFloat))
    val weights = seq.getWeightsBias()
    weights.sameElements(Array[Float]()) should be (true)

    // validate java serialization
    val seq2 = seq.cloneModule()
    val output2 = seq2.forward(input).toTensor[Float].clone()
    output should be (output2)

    // validate kryo serialization
    val serde = new KryoSerializer(new SparkConf()).newInstance()
    val buff = serde.serialize(seq)
    val seq3 = serde.deserialize[Sequential[Float]](buff)
    val output3 = seq3.forward(input).toTensor[Float].clone()
    output should be (output3)
  }

  "WordEmbedding GloVe without wordIndex" should "work properly" in {
    val seq = Sequential[Float]()
    val layer = WordEmbedding[Float](embeddingFile, inputLength = 1)
    seq.add(layer)
    val input = Tensor[Float](3, 1)
    input(Array(1, 1)) = 1  // "the"
    input(Array(2, 1)) = 18 // "with"
    input(Array(3, 1)) = 0  // unknown word
    val output = seq.forward(input).toTensor[Float]
    output.size() should be (Array(3, 1, 50))
    val expected = Array(0.418, 0.24968, -0.41242, 0.1217, 0.34527,
      -0.044457, -0.49688, -0.17862, -0.00066023, -0.6566,
      0.27843, -0.14767, -0.55677, 0.14658, -0.0095095,
      0.011658, 0.10204, -0.12792, -0.8443, -0.12181,
      -0.016801, -0.33279, -0.1552, -0.23131, -0.19181,
      -1.8823, -0.76746, 0.099051, -0.42125, -0.19526,
      4.0071, -0.18594, -0.52287, -0.31681, 0.00059213,
      0.0074449, 0.17778, -0.15897, 0.012041, -0.054223,
      -0.29871, -0.15749, -0.34758, -0.045637, -0.44251,
      0.18785, 0.0027849, -0.18411, -0.11514, -0.78581,
      0.25616, 0.43694, -0.11889, 0.20345, 0.41959,
      0.85863, -0.60344, -0.31835, -0.6718, 0.003984,
      -0.075159, 0.11043, -0.73534, 0.27436, 0.054015,
      -0.23828, -0.13767, 0.011573, -0.46623, -0.55233,
      0.083317, 0.55938, 0.51903, -0.27065, -0.28211,
      -1.3918, 0.17498, 0.26586, 0.061449, -0.273,
      3.9032, 0.38169, -0.056009, -0.004425, 0.24033,
      0.30675, -0.12638, 0.33436, 0.075485, -0.036218,
      0.13691, 0.37762, -0.12159, -0.13808, 0.19505,
      0.22793, -0.17304, -0.07573, -0.25868, -0.39339,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0)
    output.storage().toArray should be (expected.map(_.toFloat))
    val weights = seq.getWeightsBias()
    weights.sameElements(Array[Float]()) should be (true)
    val wordIndex = WordEmbedding.getWordIndex(embeddingFile)
    wordIndex.size should be (21)
    wordIndex("for") should be (11)
    wordIndex("of") should be (4)
    wordIndex("it") should be (21)
  }

}

class WordEmbeddingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val gloveDir = getClass.getClassLoader.getResource("glove.6B").getPath
    val embeddingFile = gloveDir + "/glove.6B.50d.txt"
    val layer = WordEmbedding[Float](embeddingFile, inputLength = 1)
    layer.build(Shape(4, 1))
    val input = Tensor[Float](4, 1)
    input(Array(1, 1)) = 5
    input(Array(2, 1)) = 10
    input(Array(3, 1)) = 0
    input(Array(4, 1)) = 12
    runSerializationTest(layer, input)
  }
}
