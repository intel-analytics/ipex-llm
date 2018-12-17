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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

@com.intel.analytics.bigdl.tags.Parallel
class LookupTableSparseSpec extends FlatSpec with Matchers {
  "A LookupTableSparse without weight" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sum")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(input)
    val exceptedOutput = Tensor(3, 4)
    exceptedOutput.select(1, 1).range(18, 24, 2)
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(5, 8)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01f, 6.01f, 7.01f, 8.01f
    )))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      10.101f, 12.101f, 14.101f, 16.101f
    )))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f
    )))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse mean without weight" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "mean")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(input)
    val exceptedOutput = Tensor(3, 4)
    exceptedOutput.select(1, 1).range(9, 12)
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(5, 8)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01f, 6.01f, 7.01f, 8.01f
    )))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      9.551f, 11.051f, 12.551f, 14.051f
    )))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      0.55f, 1.05f, 1.55f, 2.05f
    )))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse sqrtn without weight" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sqrtn")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(input)
    val exceptedOutput = Tensor(3, 4)
    exceptedOutput.select(1, 1).copy(Tensor(T(
      12.72792244f, 14.14213562f, 15.55634975f, 16.97056389f
    )))
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(5, 8)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01f, 6.01f, 7.01f, 8.01f
    )))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      9.77881813f, 11.48592472f, 13.19303131f, 14.9001379f
    )))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      0.77781749f, 1.4849242f, 2.19203091f, 2.89913774f
    )))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "mean")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(3, 4)
    exceptedOutput.select(1, 1).range(66, 96, 10).div(10)
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(5, 8)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01f, 6.01f, 7.01f, 8.01f
    )))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      9.881f, 11.681f, 13.481f, 15.281f
    )))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      0.22f, 0.42f, 0.62f, 0.82f
    )))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse sum" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sum")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(3, 4)
    exceptedOutput.select(1, 1).range(33, 48, 5).div(2)
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(15, 24, 3)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01000023f, 6.01000023f, 7.01000023f, 8.01000023f)))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      29.20300217f, 34.20300217f, 39.20300217f, 44.20300217f)))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      0.55000001f, 1.04999995f, 1.54999995f, 2.04999995f)))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse sqrtn" should "generate correct output" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sqrtn")
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(3, 4)
    // this result copy from tensorflow
    exceptedOutput.select(1, 1).copy(Tensor(T(
      8.00367546f, 9.21635437f, 10.42903233f, 11.64171028f)))
    exceptedOutput.select(1, 2).range(1, 4)
    exceptedOutput.select(1, 3).range(5, 8)
    output should be (exceptedOutput)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      5.01000023f, 6.01000023f, 7.01000023f, 8.01000023f)))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      10.06815679f, 12.03829916f, 14.00844176f, 15.97858436f)))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      0.2667892f, 0.50932479f, 0.75186044f, 0.99439609f)))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse sum with norm2" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sum", maxNorm = 2)
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(T(
     1.96314502f, 2.30076504f, 2.63838482f, 2.9760046f,
     0.36514834f, 0.73029667f, 1.09544504f, 1.46059334f,
     2.27429366f, 2.72915244f, 3.18401146f, 3.63887f)).resize(3, 4)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      9.76163447e-01f, 4.88081932e-01f, 5.9604645e-08f, -4.88081217e-01f)))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      0.1611459979f, 0.065922181f, -0.0292999346f, -0.124521899f)))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      -4.44917269e-02f, -1.64425969e-02f, 1.16065294e-02, 3.96556593e-02)))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse mean with norm2" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "mean", maxNorm = 2)
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(T(
      0.785258f, 0.92030603f, 1.05535388f, 1.19040179,
      0.36514834f, 0.73029667f, 1.09544504f, 1.46059334,
      0.75809789f, 0.9097175f, 1.06133711f, 1.21295667)).resize(3, 4)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      9.76163447e-01f, 4.88081932e-01f, 5.9604645e-08f, -4.88081217e-01f)))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      0.0337786f, 0.0138183197f, -0.0061413685f, -0.0261015609f)))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      -1.77966896e-02f, -6.57703914e-03f, 4.64261323e-03f, 1.58622656e-02f)))

    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

  "A LookupTableSparse sqrtn with norm2" should "generate correct output and gradient" in {
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val layer1 = LookupTableSparse(10, 4, "sqrtn", maxNorm = 2)
    layer1.weight.range(1, 40, 1)
    val output = layer1.forward(T(input, weight))
    val exceptedOutput = Tensor(T(
      0.9522652f, 1.11603498f, 1.27980471f, 1.44357431,
      0.36514834f, 0.73029667f, 1.09544504f, 1.46059334,
      0.75809789f, 0.9097175f, 1.06133711f, 1.21295667)).resize(3, 4)

    val gradOutput = Tensor(T(
      1.1f, 2.1f, 3.1f, 4.1f,
      5.01f, 6.01f, 7.01f, 8.01f,
      9.001f, 10.001f, 11.001f, 12.001f
    )).resizeAs(output)

    layer1.backward(input, gradOutput)

    val exceptedGradWeight = Tensor(10, 4)
    exceptedGradWeight.select(1, 1).copy(Tensor(T(
      9.76163447e-01f, 4.88081932e-01f, 5.9604645e-08f, -4.88081217e-01f)))
    exceptedGradWeight.select(1, 2).copy(Tensor(T(
      0.008337097f, 0.0034106036f, -0.0015158607f, -0.006442174f)))
    exceptedGradWeight.select(1, 4).copy(Tensor(T(
      -2.15816516e-02f, -7.97582790e-03f, 5.63000515e-03f, 1.92358345e-02f)))

    print(output)
    output should be (exceptedOutput)
    layer1.gradWeight should be (exceptedGradWeight)
  }

}

class LookupTableSparseSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val lookupTableSparse = LookupTableSparse[Float](20, 10, "sum", 1)
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse[Float](Array(indices1, indices2), values, Array(3, 4))
    runSerializationTest(lookupTableSparse, input, lookupTableSparse.getClass)
  }
}
