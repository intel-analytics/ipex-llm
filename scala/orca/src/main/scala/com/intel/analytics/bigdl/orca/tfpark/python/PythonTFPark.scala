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
package com.intel.analytics.zoo.tfpark.python

import java.nio.FloatBuffer
import java.util.{List => JList}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.python.api.{JTensor, Sample}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.{PythonZoo, RDDWrapper}
import com.intel.analytics.zoo.tfpark._
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.apache.spark.rdd.RDD
import org.tensorflow.{Graph, Session, Tensor => TTensor}

import scala.reflect.ClassTag


object PythonTFPark {

  def ofFloat(): PythonTFPark[Float] = new PythonTFPark[Float]()

  def ofDouble(): PythonTFPark[Double] = new PythonTFPark[Double]()

}


class PythonTFPark[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {

  def createTFTrainingHelper(modelPath: String, config: Array[Byte] = null): Module[Float] = {
    TFTrainingHelper(modelPath, config)
  }

  def saveCheckpoint(model: TFTrainingHelper): Unit = {
    model.saveCheckpoint()
  }

  def createIdentityCriterion(): IdentityCriterion = {
    new IdentityCriterion()
  }

  def createMergeFeatureLabelImagePreprocessing(): MergeFeatureLabel = {
    new MergeFeatureLabel()
  }

  def createMergeFeatureLabelFeatureTransformer(): MergeFeatureLabelFeatureTransformer = {
    new MergeFeatureLabelFeatureTransformer()
  }

  def createTFValidationMethod(valMethod: ValidationMethod[Float], name: String,
                               outputIndices: java.util.List[Int],
                               labelIndices: java.util.List[Int]): TFValidationMethod = {
    new TFValidationMethod(valMethod, name, outputIndices, labelIndices)
  }

  def createStatelessMetric(name: String, idx: Int): StatelessMetric = {
    new StatelessMetric(name, idx)
  }

  def createGanOptimMethod(dOptim: OptimMethod[T],
                           gOptim: OptimMethod[T],
                           dStep: Int, gStep: Int, gParamSize: Int): OptimMethod[T] = {
    new GanOptimMethod[T](dOptim, gOptim, dStep, gStep, gParamSize)
  }



  def createMiniBatchRDDFromStringRDD(stringRDD: JavaRDD[Array[Byte]],
                                      batchSize: Int): RDDWrapper[StringMiniBatch[T]] = {
    import TFTensorNumeric.NumericByteArray

    val rdd = stringRDD.rdd.mapPartitions { stringIter =>
      stringIter.grouped(batchSize).map { data =>
        val tensor = Tensor[Array[Byte]](data.toArray, shape = Array(data.length))
        new StringMiniBatch[T](tensor)
      }
    }
    RDDWrapper[StringMiniBatch[T]](rdd)
  }

  def createRDDFromTFRecords(path: String,
                             jsc: JavaSparkContext,
                             serializedParseGraph: Array[Byte],
                             inputName: String,
                             outputNames: JList[String]): RDD[Sample] = {
    val sc = jsc.sc

    val bserializedParseGraph = sc.broadcast(serializedParseGraph)
    val sampleRdd = sc.newAPIHadoopFile[org.apache.hadoop.io.BytesWritable,
      org.apache.hadoop.io.NullWritable,
      org.tensorflow.hadoop.io.TFRecordFileInputFormat](path).map { KV =>
      KV._1.copyBytes()
    }.mapPartitions { iter =>
      val graphDef = bserializedParseGraph.value
      val g = new Graph()
      g.importGraphDef(graphDef)
      val sess = new Session(g)

      def addFetches(names: JList[String], runner: Session#Runner) = {
        var j = 0
        while (j < names.size()) {
          runner.fetch(names.get(j))
          j += 1
        }
      }

      def getFetches(results: JList[TTensor[_]]) = {
        val tensors = new java.util.ArrayList[JTensor](results.size())
        var j = 0
        while (j < results.size()) {
          val t = results.get(j)
          tensors.add(tfTensor2JTensor(t))
          j += 1
        }
        tensors
      }


      val records = iter.toArray
      val samples = new Array[Sample](records.length)
      var i = 0

      while (i < records.length) {

        val bytes = records(i)
        val input = TTensor.create(bytes)
        val runner = sess.runner()
        runner.feed(inputName, input)
        addFetches(outputNames, runner)
        val results = runner.run()
        val outputTensors = getFetches(results)

        input.close()
        var j = 0
        while (j < results.size()) {
          results.get(j).close()
          j += 1
        }

        samples(i) = Sample(outputTensors, new java.util.ArrayList[JTensor], "float")

        i += 1
      }

      sess.close()
      g.close()

      samples.toIterator
    }

    sampleRdd
  }

  private def tfTensor2JTensor(t: TTensor[_]): JTensor = {
    val shape = t.shape().map(_.toInt)
    val length = shape.product
    val data = new Array[Float](length)
    val buffer = FloatBuffer.wrap(
      data,
      0,
      length)
    t.writeTo(buffer)
    JTensor(data, shape, "float")
  }

}
