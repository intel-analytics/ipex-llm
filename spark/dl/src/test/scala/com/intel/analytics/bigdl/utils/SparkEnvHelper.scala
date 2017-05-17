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

package org.apache.spark

import com.intel.analytics.bigdl.utils.{Engine, TestUtils => BigDLTestUtils}
import org.mockito.Mockito._
import org.scalatest.mock.MockitoSugar


object SparkEnvHelper extends MockitoSugar {

  val scMock = mock[SparkContext]
  when(scMock.getConf).thenReturn(Engine.createSparkConf().setAppName("EngineSpecTest"))
  SparkContext.setActiveContext(scMock, false)

  /**
   * Set envs for spark local mode
   */
  def sparkLocalEnv[T](
                        core: Int = 4
                      )(body: => T): Unit = {
    when(scMock.master).thenReturn(s"local[$core]")
    System.setProperty("SPARK_SUBMIT", "true")
    body
    System.clearProperty("SPARK_SUBMIT")
    System.clearProperty("spark.master")
  }

  /**
   * Set envs for spark standalone mode
   */
  def sparkStandaloneEnv[T](
                             totalCore : Int,
                             core : Int
                           )(body : => T): Unit = {
    System.setProperty("SPARK_SUBMIT", "true")
    when(scMock.master).thenReturn(s"spark://host:7077")
    System.setProperty("spark.cores.max", totalCore.toString)
    System.setProperty("spark.executor.cores", core.toString)
    body
    System.clearProperty("SPARK_SUBMIT")
    System.clearProperty("spark.master")
    System.clearProperty("spark.cores.max")
    System.clearProperty("spark.executor.cores")
  }

  /**
   * Set envs for spark yarn mode
   */
  def sparkYarnEnv[T](
                       executors : Int,
                       core : Int
                     )(body : => T): Unit = {
    System.setProperty("SPARK_SUBMIT", "true")
    System.setProperty("spark.master", s"yarn")
    when(scMock.master).thenReturn("yarn")
    System.setProperty("spark.executor.instances", executors.toString)
    System.setProperty("spark.executor.cores", core.toString)
    body
    System.clearProperty("SPARK_SUBMIT")
    System.clearProperty("spark.master")
    System.clearProperty("spark.executor.instances")
    System.clearProperty("spark.executor.cores")
  }

  /**
   * Set envs for mesos yarn mode
   */
  def sparkMesosEnv[T](
                        totalCore : Int,
                        core : Int
                      )(body : => T): Unit = {
    System.setProperty("SPARK_SUBMIT", "true")
    when(scMock.master).thenReturn("mesos")
    System.setProperty("spark.cores.max", totalCore.toString)
    System.setProperty("spark.executor.cores", core.toString)
    body
    System.clearProperty("SPARK_SUBMIT")
    System.clearProperty("spark.master")
    System.clearProperty("spark.cores.max")
    System.clearProperty("spark.executor.cores")
  }
}