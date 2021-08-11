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

package com.intel.analytics.bigdl.nn.mkldnn

import java.io.{File, PrintWriter}
import java.nio.channels.FileChannel
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}

import breeze.numerics.abs
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.mkldnn.Phase.TrainingPhase
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Storage, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.sys.process._

object Tools {
  def error[@specialized(Float, Double) T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var ret = 0.0
    val storage1 = tensor1.storage().array()
    val storage2 = tensor2.storage().array()
    for (i <- 0 until tensor1.nElement()) {
      ret += math.abs(
        ev.toType[Double](storage1(i)) - ev.toType[Double](storage2(i)))
    }
    ret
  }

  def cumulativeError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    val ret = error[T](tensor1, tensor2)
    println((msg, "CUMULATIVE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() > 0)
    val ret = error[T](tensor1, tensor2) / tensor1.nElement()
    println((msg, "AVERAGE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](m1: Map[String, Tensor[T]],
                                m2: Map[String, Tensor[T]],
                                err: Map[String, Double])(implicit ev: TensorNumeric[T]): Unit = {
    require(m1.keySet == m2.keySet)
    require(m1.keySet subsetOf err.keySet)

    m1.keySet.foreach(i => {
      val err = error(m1(i), m2(i)) / m1(i).nElement()
      printf("%20s = %E\n", i.toUpperCase(), err)
    })
  }

  def averageAllTensors[T: ClassTag](tensor1: Tensor[T], msg: String = "Unknown")(
      implicit ev: TensorNumeric[T]): Unit = {
    val sum = tensor1.storage().array().foldLeft(ev.fromType[Int](0))((l, r) => ev.plus(l, r))
    val num = ev.fromType[Int](tensor1.nElement())
    println(("AVERGE", msg, ev.divide(sum, num)).productIterator.mkString(" ").toUpperCase())
  }

  def printTensor[T: ClassTag](tensor: Tensor[T], num: Int = 16, msg: String = "Unknown")(
      implicit ev: TensorNumeric[T]): Unit = {
    println(msg.toUpperCase)
    for (i <- 0 until num) {
      println((i, ev.toType[Double](tensor.storage().array()(i))).productIterator.mkString("\t"))
    }
  }

  private def fileName(name: String, identity: String): String = {
    val tmpdir = System.getProperty("java.io.tmpdir")
    val dirname = if (tmpdir.endsWith("/")) {
      tmpdir
    } else {
      tmpdir + "/"
    }

    val filename = if (identity.isEmpty) {
      ".bin"
    } else {
      "." + identity + ".bin"
    }

    dirname + name + filename
  }

  /*
   * @brief read binary in tmp dir to Tensor, which is used for comparing
   *        with Intel Caffe with MKL-DNN
   */
  def getTensor(name: String, size: Array[Int], identity: String): Tensor[Float] = {
    val tensor = Tensor[Float]()
    val file = fileName(name, identity)

    if (Files.exists(Paths.get(file))) {
      println(s"[INFO] load $file")
      setTensorFloat()

      def loadData(name: String): ByteBuffer = {
        val fileChannel: FileChannel = Files.newByteChannel(
          Paths.get(name),
          StandardOpenOption.READ,
          StandardOpenOption.DELETE_ON_CLOSE).asInstanceOf[FileChannel]
        val byteBuffer: ByteBuffer = ByteBuffer.allocate(fileChannel.size().toInt)
        byteBuffer.order(ByteOrder.nativeOrder())
        fileChannel.read(byteBuffer)
        byteBuffer.flip()
        byteBuffer
      }


      def setTensorFloat(): Unit = {
        val data = loadData(file).asFloatBuffer()
        val array = new Array[Float](data.limit())
        data.get(array)
        assert(size.product == array.length, s"the data length is not correct")
        tensor.set(Storage(array), sizes = size)
      }
    }

    tensor
  }

  def flattenModules(model: Module[Float], modules: ArrayBuffer[Module[Float]]): Unit = {
    model match {
      case container : Container[_, _, _] =>
        if (container.modules.nonEmpty) {
          for (i <- container.modules) {
            flattenModules(i.asInstanceOf[Module[Float]], modules)
          }
        }
      case x => if (!x.isInstanceOf[ReorderMemory] && !x.isInstanceOf[Identity]) {
        modules += model
      }
    }
  }

  def randTimes(): Int = 10

  def loadWeights(module: Module[Float], identity: String): Unit = {
    val params = module.parameters()._1
    val name = module.getName()
    module match {
      case bn: SpatialBatchNormalization =>
        val channel = bn.weightAndBias.size(1) / 2

        val weight = Tools.getTensor(s"Fwrd_${bn.getName}.Wght.3", Array(channel), identity)
        val bias = Tools.getTensor(s"Fwrd_${bn.getName}.Wght.4", Array(channel), identity)
        val weightAndBias = Tensor[Float].resize(Array(2, channel))
        if (weight.isEmpty) {weight.resize(Array(channel)).fill(1)}
        weightAndBias.select(1, 1).copy(weight)
        if (bias.isEmpty) {
          bias.resize(Array(channel)).fill(0)
        }
        weightAndBias.select(1, 2).copy(bias)
        bn.weightAndBias.copy(weightAndBias.view(bn.weightAndBias.size()))
      case _ =>
        for (j <- params.indices) {
          val w = Tools.getTensor(s"Fwrd_$name.Wght.$j", params(j).size(), identity)
          module match {
            case layer: MklDnnLayer =>
              val weights = if (!w.isEmpty) {
                params(j).copy(w)
              } else {
                val zeros = Tensor[Float]().resize(params(j).size()).fill(0)
                params(j).copy(zeros)
              }
            case _ =>
              params(j).copy(w)
          }
        }
    }
  }

  def compareGradients(module: Module[Float], epsilon: Float, identity: String): Boolean = {
    var ret = true

    val name = module.getName()
    val params = module.parameters()._2

    module match {
      case bn: SpatialBatchNormalization =>
        val channel = bn.weightAndBias.size(1) / 2

        val weight = Tools.getTensor(s"Bwrd_${bn.getName}.Grad.3", Array(channel), identity)
        val bias = Tools.getTensor(s"Bwrd_${bn.getName}.Grad.4", Array(channel), identity)
        val weightAndBias = Tensor[Float].resize(Array(2, channel))
        weightAndBias.select(1, 1).copy(weight)
        weightAndBias.select(1, 2).copy(bias)

        ret &= Equivalent.nearequals(weightAndBias.view(bn.gradWeightAndBias.size()),
          dense(bn.gradWeightAndBias.native).toTensor, epsilon)
        val runningMean = Tools.getTensor(s"Fwrd_$name.Wght.0", Array(channel), identity)
        val runningVariance = Tools.getTensor(s"Fwrd_$name.Wght.1", Array(channel), identity)

        ret &= compare2Tensors(runningMean, dense(bn.runningMean.native).toTensor)
        ret &= compare2Tensors(runningVariance, dense(bn.runningVariance.native).toTensor)

        assert(ret, s"${module.getName()} gradient can't pass, please check")
      case _ =>
        for (j <- params.indices) {
          val w = Tools.getTensor(s"Bwrd_$name.Grad.$j", params(j).size(), identity)
          ret &= Equivalent.nearequals(params(j), w, epsilon)

          assert(ret, s"${module.getName()} gradient $j can't pass, please check")
        }
    }

    ret
  }

  def compare(prototxt: String, model: Module[Float], inputShape: Array[Int],
    outputShape: Array[Int], epsilon: Double = 1e-7): Unit = {
    val identity = Collect.run(prototxt, singleLayer = true)
    val modules = ArrayBuffer[Module[Float]]()
    Tools.flattenModules(model, modules)

    val input = Tools.getTensor("Fwrd_data", inputShape, identity)
    val gradOutput = Tools.getTensor(s"Bwrd_${modules.last.getName()}.loss", outputShape, identity)

    modules.filter(_.parameters() != null).foreach(loadWeights(_, identity))

    model.forward(input)
    model.backward(input, gradOutput)

    for (i <- modules.indices) {
      compareSingleLayer(modules(i), identity)
    }

    def compareSingleLayer(module: Module[Float], identity: String): Boolean = {
      val name = module.getName()
      val bigdlOutput = module.output.toTensor[Float]
      val bigdlGradInput = if (module.isInstanceOf[CAddTable]) {
        module.gradInput.toTable.apply[Tensor[Float]](1)
      } else {
        module.gradInput.toTensor[Float]
      }

      val noPaddingOutputShape = if (module.isInstanceOf[MklDnnModule]) {
        module.asInstanceOf[MklDnnModule].outputFormats()(0).shape
      } else {
        bigdlOutput.size()
      }

      val noPaddingGradInputShape = if (module.isInstanceOf[MklDnnModule]) {
        module.asInstanceOf[MklDnnModule].gradInputFormats()(0).shape
      } else {
        bigdlGradInput.size()
      }

      val output = Tools.getTensor(s"Fwrd_$name", noPaddingOutputShape, identity)
      val gradInput = Tools.getTensor(s"Bwrd_$name", noPaddingGradInputShape, identity)

      var ret = true

      module match {
        case layer: MklDnnLayer =>
          ret &= compare2Tensors(output, toNCHW(bigdlOutput, layer.outputFormats()(0)))
          assert(ret, s"${module.getName()} output can't pass, please check")

          ret &= compare2Tensors(gradInput, toNCHW(bigdlGradInput, layer.gradInputFormats()(0)))
          assert(ret, s"${module.getName()} gradInput can't pass, please check")
        case _ =>
          ret &= compare2Tensors(output, bigdlOutput)
          assert(ret, s"${module.getName()} output can't pass, please check")

          ret &= compare2Tensors(gradInput, bigdlGradInput)
          assert(ret, s"${module.getName()} gradInput can't pass, please check")
      }

      if (module.parameters() == null) {
        return ret
      }

      val params = module.parameters()._2
      compareGradients(module, epsilon.toFloat, identity)

      ret
    }
  }

  def compare2Tensors(src: Tensor[Float], dst: Tensor[Float]): Boolean = {
    Equivalent.nearequals(dense(src).toTensor, dense(dst).toTensor)
  }

  def dense(t: Activity): Activity = {
    val ret = if (t.isTensor) {
      val tt = t.asInstanceOf[Tensor[Float]]
      Tensor[Float]().resize(tt.size()).copy(tt)
    } else {
      throw new UnsupportedOperationException
    }

    ret
  }

  def toNCHW(src: Tensor[Float], inputFormat: MemoryData): Tensor[Float] = {
    val outputFormat = HeapData(inputFormat.shape,
      if (src.size().length == 2) { Memory.Format.nc } else { Memory.Format.nchw })
    val reorder = ReorderMemory.create(inputFormat, outputFormat, null, null)

    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.forward(src).toTensor
  }

  def fromNCHW(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = src.size().length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.nc
      case 4 => Memory.Format.nchw
    }

    val inputFormat = HeapData(src.size(), defaultFormat)
    val reorder = ReorderMemory.create(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.forward(src).toTensor
  }

  def fromOIHW(src: Tensor[Float], outputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = outputFormat.shape.length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
    }

    val inputFormat = HeapData(outputFormat.shape, defaultFormat)
    val reorder = ReorderMemory.create(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.updateOutput(src).toTensor
  }

  def toOIHW(src: Tensor[Float], inputFormat: MemoryData): Tensor[Float] = {
    val defaultFormat = inputFormat.shape.length match {
      case 1 => Memory.Format.x
      case 2 => Memory.Format.oi
      case 4 => Memory.Format.oihw
      case 5 => Memory.Format.goihw
    }

    val outputFormat = HeapData(inputFormat.shape, defaultFormat)
    val reorder = ReorderMemory.create(inputFormat, outputFormat, null, null)
    reorder.setRuntime(new MklDnnRuntime)
    reorder.initFwdPrimitives(Array(inputFormat), TrainingPhase)
    reorder.updateOutput(src).toTensor
  }
}

/**
 * Call "collect" command, which is a method to collect output binary files.
 * It's similar to "caffe collect", the difference is that it supports collect
 * single layers output and gradient through make a fake gradOutput/top_diff.
 */
object Collect {
  val tmpdir: String = System.getProperty("java.io.tmpdir")
  val collectPath: String = System.getProperty("collect.location")

  def hasCollect: Boolean = {
    val exitValue = if (collectPath != null) s"ls $collectPath".! else "which collect".!
    exitValue == 0
  }

  /**
   * save the prototxt to a temporary file and call collect
   * @param prototxt prototxt with string
   * @return the middle random number in temporary file, which is an identity for getTensor.
   */
  def run(prototxt: String, singleLayer: Boolean = true): String = {
    def saveToFile(prototxt: String, name: String): String = {
      val tmpFile = java.io.File.createTempFile(name, ".prototxt")
      val absolutePath = tmpFile.getAbsolutePath

      println(s"prototxt is saved to $absolutePath")

      val writer = new PrintWriter(tmpFile)
      writer.println(prototxt)
      writer.close()

      absolutePath
    }

    if (! hasCollect) {
      throw new RuntimeException(s"Can't find collect command. Have you copy to the PATH?")
    }

    val file = saveToFile(prototxt, "UnitTest.") // UnitTest ends with dot for getting random number
    val identity = file.split("""\.""").reverse(1) // get the random number

    val cmd = Seq(s"$collectPath", "--model", file, "--type", "float", "--identity", identity)
    val exitValue = if (singleLayer) {
      Process(cmd :+ "--single", new File(tmpdir)).!
    } else {
      Process(cmd, new File(tmpdir)).!
    }

    Files.deleteIfExists(Paths.get(file))
    require(exitValue == 0, s"Something wrong with collect command. Please check it.")

    identity
  }
}

object TestUtils {
  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val takes = (t1 - t0) / 1e9
    (takes, result)
  }

  def manyTimes[R](block: => R)(iters: Int): (Double, R) = {
    time[R] {
      var i = 0
      while (i < iters - 1) {
        block
        i += 1
      }
      block
    }
  }

  def speedup(base: Double, after: Double): String = {
    val result = (base - after) / base
    ((result * 1000).toInt / 10.0).toString + "%"
  }
}

object Equivalent {

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
    epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    result
  }

  def getunequals(t1: Tensor[Float], t2: Tensor[Float],
    epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    var num = 0
    t1.map(t2, (a, b) => {
      if (true) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          num += 1
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    println("diff num " + num)
    return true
  }

  def isEquals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = if (a == b) true else false
        if (!result) {
          val diff = Math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return result
  }
}
