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
package com.intel.analytics.bigdl.utils.tf

import com.intel.analytics.bigdl.utils.TestUtils.processPath
import java.io.{OutputStream, File => JFile}
import java.nio.ByteOrder

import com.google.protobuf.CodedOutputStream
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildCard, TensorNumeric}
import com.intel.analytics.bigdl.utils.{BigDLSpecHelper, FileWriter, RandomGenerator, T}
import com.intel.analytics.bigdl.utils.tf.Tensorflow.const
import org.tensorflow.framework.{GraphDef, NodeDef}

import scala.language.postfixOps
import scala.reflect.ClassTag
import scala.sys.process._
import scala.util.control.NonFatal

abstract class TensorflowSpecHelper extends BigDLSpecHelper {

  protected def tfCheck(): Unit = {
    var exitValue : String = ""
    try {
      exitValue = ((Seq("python", "-c", "import sys; print ','.join(sys.path)"))!!)
      ((Seq("python", "-c", "import tensorflow"))!!)
    } catch {
      case NonFatal(e) => cancel("python or tensorflow is not installed", e)
    }

    if (!exitValue.contains("models")) {
      cancel("Tensorflow models path is not exported")
    }
  }

  protected def runPython(cmd: String): Boolean = {
    try {
      logger.info("run command\n" + cmd)
      val proc = s"python $cmd".run
      return proc.exitValue() == 0
    } catch {
      case NonFatal(e) => false
    }
  }

  protected def runPythonSaveTest(graphPath: String, outputSuffix: String) : Boolean = {
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      s"save_test.py $graphPath $outputSuffix"
    runPython(path)
  }


  /**
   * Compare the output from tf operation and BigDL
   * @param nodeDefBuilder
   * @param inputs
   * @param outputIndex start from 0
   * @param delta error tolerant
   */
  protected def compare[T: ClassTag](nodeDefBuilder: NodeDef.Builder,
                           inputs: Seq[Tensor[_]], outputIndex: Int,
      delta: Double = 1e-5)(implicit ev: TensorNumeric[T])
  : Unit = {
    val graphFile = saveGraph(nodeDefBuilder, inputs)
    val bigdlOutput = runGraphBigDL[T](graphFile, nodeDefBuilder.getName)
    val bigdlOutputTensor = if (bigdlOutput.isTensor) {
      require(outputIndex == 0, s"invalid output index $outputIndex")
      bigdlOutput.asInstanceOf[Tensor[_]]
    } else {
      bigdlOutput.toTable.apply[Tensor[_]](outputIndex + 1)
    }
    val tfOutput = runGraphTF[T](graphFile, nodeDefBuilder.getName + s":$outputIndex")
    bigdlOutputTensor.asInstanceOf[Tensor[NumericWildCard]]
      .almostEqual(tfOutput.asInstanceOf[Tensor[NumericWildCard]], delta) should be(true)
  }

  protected def compare[T: ClassTag](nodeDefBuilder: NodeDef.Builder,
    inputs: Seq[Tensor[_]], outputIndexes: Seq[Int],
    delta: Double)(implicit ev: TensorNumeric[T])
  : Unit = {
    outputIndexes.foreach(compare(nodeDefBuilder.clone(), inputs, _, delta))
  }

  protected def getResult[T: ClassTag, D](nodeDefBuilder: NodeDef.Builder, inputs: Seq[Tensor[_]],
    outputIndex: Int)(implicit ev: TensorNumeric[T]): (Tensor[D], Tensor[D]) = {
    val graphFile = saveGraph(nodeDefBuilder, inputs)
    val bigdlOutput = runGraphBigDL[T](graphFile, nodeDefBuilder.getName)
    val bigdlOutputTensor = if (bigdlOutput.isTensor) {
      require(outputIndex == 0, s"invalid output index $outputIndex")
      bigdlOutput.asInstanceOf[Tensor[D]]
    } else {
      bigdlOutput.toTable.apply[Tensor[D]](outputIndex + 1)
    }
    val tfOutput = runGraphTF(graphFile, nodeDefBuilder.getName + s":$outputIndex")
    (bigdlOutputTensor, tfOutput.asInstanceOf[Tensor[D]])
  }

  private def saveGraph(nodeDefBuilder: NodeDef.Builder, inputs: Seq[Tensor[_]]): String = {
    var i = 0
    val inputConsts = inputs.map(input => {
      i += 1
      const(input, s"TensorflowLoaderSpecInput_$i", ByteOrder.LITTLE_ENDIAN)
    })
    inputConsts.foreach(p => nodeDefBuilder.addInput(p.getName))

    val graphBuilder = GraphDef.newBuilder()
    graphBuilder.addNode(nodeDefBuilder.build())
    inputConsts.foreach(graphBuilder.addNode(_))

    var fw: FileWriter = null
    var out: OutputStream = null
    try {
      val file = createTmpFile()
      fw = FileWriter(file.getAbsolutePath)
      out = fw.create(true)
      val output = CodedOutputStream.newInstance(out)
      val graph = graphBuilder.build()
      graph.writeTo(output)
      output.flush()
      out.flush()
      file.getAbsolutePath
    } finally {
      if (out != null) out.close()
      if (fw != null) fw.close()
    }
  }

  private def runGraphBigDL[T: ClassTag](graph: String, output: String)
                                        (implicit ev: TensorNumeric[T]): Activity = {
    val m = Module.loadTF[T](graph, Seq(), Seq(output))
    m.forward(null)
  }

  private def runGraphTF[T: ClassTag](graph: String, output: String)
                                     (implicit ev: TensorNumeric[T]): Tensor[_] = {
    tfCheck()
    val outputFile = createTmpFile()
    val outputFolder = getFileFolder(outputFile.getAbsolutePath())
    val outputFileName = getFileName(outputFile.getAbsolutePath())
    val resource = getClass().getClassLoader().getResource("tf")
    val path = processPath(resource.getPath()) + JFile.separator +
      s"run-graph.py $graph $output $outputFolder $outputFileName result"
    runPython(path)
    val m = Module.loadTF[T](outputFile.getAbsolutePath, Seq(), Seq("result"))
    m.forward(null).asInstanceOf[Tensor[_]]
  }
}
