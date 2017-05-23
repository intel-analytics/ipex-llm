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

package org.apache.spark.scheduler.cluster.mesos

import org.apache.spark.{SparkConf, SparkContext}

object MesosResources {
  /**
   * Get the totalCoreCount, which will be updated in RPC method (receiveAndReply),
   *
   * Because it's a protected attribute, we can't get it directly. We get it through
   * `defaultParallism()` because it will call `totalCoreCount.get()` if the property
   * `spark.default.parallelism` has not been set.
   *
   * @param conf spark config
   * @param backend mesos coarse backend
   * @return current totalCoreCount
   */
  private def getTotalCoreCount(conf: SparkConf,
    backend: MesosCoarseGrainedSchedulerBackend): Int = {
    val parallismOption = "spark.default.parallelism"
    val containOption = conf.contains(parallismOption)

    val defaultParallism = backend.defaultParallelism().toString

    if (containOption) {
      conf.remove(parallismOption)
    }

    val totalCoreCount = backend.defaultParallelism()

    if (containOption) {
      conf.set(parallismOption, defaultParallism)
    }

    totalCoreCount
  }

  def checkAllExecutorStarted(sc: SparkContext): Boolean = {
    sc.schedulerBackend match {
      case b: MesosCoarseGrainedSchedulerBackend =>
        // maxCores comes from MesosCoarseGrainedSchedulerBackend, which is a public attribute
        val maxCores = b.maxCores
        val totalCoreCount = getTotalCoreCount(sc.conf, b)

        // If we want to have the highest performance, we should wait to all resources available.
        val minRegisteredRatio = 1.0

        val ret = if (b.sufficientResourcesRegistered() &&
          totalCoreCount >= maxCores * minRegisteredRatio) {
          true
        } else {
          false
        }

        ret
      case _ => throw new IllegalArgumentException(s"Did you set the right schduler backend?" +
        s"We only support Mesos Coarse Scheduler Backend.")
    }
  }
}
