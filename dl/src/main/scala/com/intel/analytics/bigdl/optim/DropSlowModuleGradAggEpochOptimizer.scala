/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import java.util.concurrent.{Callable, LinkedBlockingQueue, ThreadPoolExecutor, TimeUnit}

import collection.JavaConversions._
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.ps.ParameterManager
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table, Util}
import org.apache.spark.TaskContext

import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag
import org.apache.log4j.Logger

object DropSlowModuleGradAggEpochOptimizer {
  val subModuleNumber = System.getProperty(
    "com.intel.analytics.bigdl.optim.BetterGradAggEpochOptimizer.subModuleNumber",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  val lossArray = new Array[Double](subModuleNumber)
  val recordsArray = new Array[Int](subModuleNumber)
  private val logger = Logger.getLogger(getClass);
  private val maxThread = System.getProperty(
    "com.intel.analytics.bigdl.optim.BetterGradAggEpochOptimizer.maxThread",
    (Runtime.getRuntime().availableProcessors() * 50 / 2).toString()).toInt

  val pool = new ThreadPoolExecutor(maxThread, maxThread, 0L, TimeUnit.MILLISECONDS,
    new LinkedBlockingQueue[Runnable])

  val context = new ExecutionContext {
    val threadPool = pool

    def execute(runnable: Runnable) {
      threadPool.submit(runnable)
    }

    def reportFailure(t: Throwable) {}
  }

  var thread : Thread = null  
  
  var weightSyncTime = 0l  
}

class DropSlowModuleGradAggEpochOptimizer[T: ClassTag](
    @transient module: Module[Tensor[T], Tensor[T], T],
    criterion: Criterion[Tensor[T], T],
    optm: OptimMethod[T],
    pm: ParameterManager[T],
    dataSets: DataSet[_, T] with HasEpoch,
    metrics: Metrics,
    config: Table = T())
  (implicit ev: TensorNumeric[T])
  extends EpochOptimizer[T](module, criterion, optm, pm, dataSets, metrics, config) {
  
  import DropSlowModuleGradAggEpochOptimizer._
  
  private def init() = {
    val broadcast = dataSet.getSparkContext().broadcast((module, criterion))
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      val test = (0 until subModuleNumber).map { _ =>
        val localModule = broadcastModule.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val (weights, grads) = localModule.getParameters()
        CachedModel(localModule, localCriterion, weights, grads, T())}
      Iterator(test.toArray)
    }).persist()
    models.setName("modelRDD")
    logger.info("Cache models...")
    models.count()
    logger.info("Cache models... done")
    models
  }

  val multiThreadModels = init()


  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    // don't send whole Optimizer in closure
    val broadcastEV = dataSets.getSparkContext().broadcast(ev)

    val sc = dataSets.getSparkContext()
    val partitionNum = dataSets.getPartitionNum()
    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(20)
    val state = T()    
    var timeout = Long.MaxValue
    var threshold = 0l
    val dropModulePercent = 0.04
    var iteration = 0
    
    for (i <- 1 to epochNum) {
      logger.info(s"[Epoch $i/$epochNum] Train start")
      val epochStart = System.nanoTime()

      logger.info("config" + config)

      logger.info(s"[Epoch $i/$epochNum] Shuffle data")
      dataSets.reset()
      val shuffleEnd = System.nanoTime()
      var accumulateCount = 0
      logger.info(s"[Epoch $i/$epochNum] Shuffle data complete. Takes ${
        (shuffleEnd -
          epochStart) / 1e9
      }s")
      config("epoch") = i

      val driverParNum = partitionNum * subModuleNumber
      while (!dataSets.epochFinished()) {
        val lossSum = sc.accumulator(0.0, "loss sum")
        val recordsNum = sc.accumulator(0, "record number")
        val stackCount = sc.accumulator(0, "stack count")
        val dropModulesPerIter = sc.accumulator(0, "dropped modules")
        metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)
        metrics.set("init gradient time", 0.0, sc, partitionNum)
        metrics.set("construct tensor time", 0.0, sc, partitionNum)
        metrics.set("computing time average", 0.0, sc, partitionNum)
        metrics.set("prepare time", 0.0, sc, partitionNum)
        metrics.set("statics time", 0.0, sc, partitionNum)
        metrics.set("aggregate gradient time", 0.0, sc, partitionNum)
        
        val driverMetrics = metrics
        val start = System.nanoTime()
        val resultRDD = dataSets.fetch().zipPartitions(
          models,
          pm.sync(models.mapPartitions(iter => Iterator.single(iter.next().weight))),
          multiThreadModels, true)(
          (data, modelIter, weights, multiThreadModuleIter) => {
            var tmp = System.nanoTime()

            val localMTCaches = multiThreadModuleIter.next()
            val localCaches = modelIter.next()
            val syncWeightTask = Future {
              weights.next() // Update local weights
              (0 until subModuleNumber).map(i => Future {
                localMTCaches(i).weight.copy(localCaches.weight)
              }(context)).foreach(Await.result(_, Duration.Inf))
            }(context)

            val localEV = broadcastEV.value
            tmp = System.nanoTime()
            if(thread != null) {
              thread.join()
              thread = null
            }
            
            driverMetrics.add("init gradient time", System.nanoTime() - tmp)

            tmp = System.nanoTime()
            val tensorBuffer = new Array[(Tensor[T], Tensor[T])](subModuleNumber)
            val constructTensorTask = Future {
              val batch = data.next()
              var b = 0
              while(b < subModuleNumber) {
                tensorBuffer(b) = batch.next()
                b += 1
              }
            }(context)

            Await.result(constructTensorTask, Duration.Inf)
            driverMetrics.add("construct tensor time", System.nanoTime() - tmp)
            Await.result(syncWeightTask, Duration.Inf)

            driverMetrics.add("prepare time", System.nanoTime() - tmp)

            //======================Start train models===================================
            tmp = System.nanoTime()            
            val computeThreads = if(iteration<200) {
              (0 until subModuleNumber).map(i => new Callable[Int] {
                def call(): Int = {
                  val localModule = localMTCaches(i).model
                  val localCriterion = localMTCaches(i).criterion
                  val (inputFloat, targetFloat) = tensorBuffer(i)
                  val input = inputFloat.asInstanceOf[Tensor[T]]
                  val target = targetFloat.asInstanceOf[Tensor[T]]
                  val output = localModule.forward(input)
                  lossArray(i) = localEV.toType[Double](localCriterion.forward(output, target))
                  val errors = localCriterion.backward(output, target)
                  localModule.backward(input, errors)
                  recordsArray(i) = target.size(1)
                  i
                }})
            } else {
              val pid = TaskContext.getPartitionId()
              val pre = (iteration%100)*partitionNum*subModuleNumber + pid*subModuleNumber
              val tasks = (0 until subModuleNumber).map(i => new Callable[Int] {
                def call(): Int = {
                  val start = System.nanoTime()
                  val localModule = localMTCaches(i).model
                  val localCriterion = localMTCaches(i).criterion
                  val (inputFloat, targetFloat) = tensorBuffer(i)
                  val input = inputFloat.asInstanceOf[Tensor[T]]
                  val target = targetFloat.asInstanceOf[Tensor[T]]
                  val output = localModule.forward(input)
                  lossArray(i) = localEV.toType[Double](localCriterion.forward(output, target))
                  val errors = localCriterion.backward(output, target)
                  localModule.backward(input, errors)
                  recordsArray(i) = target.size(1)
                  Util.moduleTimeList(i+pre) = System.nanoTime()-start + weightSyncTime
                  i
                }
              })
              tasks
            }
            
            if(iteration > 299) {
              timeout = ((threshold-weightSyncTime)/1e6).toLong                          
            }
            val threads = pool.invokeAll(computeThreads.toSeq, timeout, TimeUnit.MILLISECONDS)
            
            val computingTime = System.nanoTime() - tmp
            driverMetrics.add("computing time average", computingTime)
            driverMetrics.add("computing time for each node", computingTime)
            
            val finishedThreads = threads.filter(!_.isCancelled).map(_.get())            
            dropModulesPerIter += (subModuleNumber-finishedThreads.size)

            
//              tmp = System.nanoTime()
              stackCount += finishedThreads.size
              finishedThreads.foreach{index =>
                lossSum += lossArray(index)
                recordsNum += recordsArray(index)
              }
//              driverMetrics.add("statics time", System.nanoTime() - tmp)

              tmp = System.nanoTime()
              val grads = localMTCaches.map(_.gradient)
              val gradLength = grads(0).nElement()
              val taskSize = gradLength / subModuleNumber
              val extraTask = gradLength % subModuleNumber

              if(finishedThreads.size>0) {
                localCaches.gradient.copy(grads(finishedThreads(0)))                
                (0 until subModuleNumber).map(tid => Future {
                  val offset = tid * taskSize + math.min(tid, extraTask)
                  val length = taskSize + (if (tid < extraTask) 1 else 0)

                  finishedThreads.drop(0).foreach{index =>
                    localCaches.gradient.narrow(1, offset + 1, length)
                      .add(grads(index).narrow(1, offset + 1, length))
                  }
                }(context)).foreach(Await.result(_, Duration.Inf))
                driverMetrics.add("aggregate gradient time", System.nanoTime() - tmp)  
              } else {
                localCaches.model.zeroGradParameters() 
              }
            
            thread = new Thread(new Runnable {
              override def run(): Unit = {
                (0 until subModuleNumber).map(i => Future {
                  localMTCaches(i).model.training()
                  localMTCaches(i).model.zeroGradParameters()
                }(context)).foreach(Await.result(_, Duration.Inf))
              }
            })
            thread.start()

            Iterator.single(localCaches.gradient)
          })
        val reduceBefore = System.nanoTime()
        val driverEV = ev
        val optM = optm
        val configDriver = config
        
        if(dropModulesPerIter.value > driverParNum*0.5) {
          logger.info("Warning!!! Ignore this iteration as more thean half module is dropped!!" +
            " Dropped module: " + dropModulesPerIter.value)
        } else {
          pm.sumAndUpdate(resultRDD, (weights, gradients, state) => {
            gradients.div(driverEV.fromType[Int](driverParNum))
            optM.optimize(_ => (driverEV.fromType(lossSum.value / stackCount.value), gradients),
              weights, configDriver, state)
          })

          accumulateCount += recordsNum.value
          val end = System.nanoTime()
          logger.info(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${
            recordsNum.value
          } in ${(end - start) / 1e9}seconds. " +
            s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
              lossSum.value / stackCount.value
            }. " +
            s"Calculate time is ${(reduceBefore - start) / 1e9}seconds. ")
          logger.info("\n" + metrics.summary())
          logger.info("\n dropModule: " + dropModulesPerIter.value)

          //compute threshold
          if(iteration%100==99 && iteration>=299) {
            Util.moduleTimeList = models.mapPartitions{ iter =>
              Util.moduleTimeList.iterator
            }.collect()
            println("size: " + Util.moduleTimeList.filter(_!=0).length)            
            val k = (dropModulePercent*100*partitionNum*subModuleNumber).toInt
            threshold = Util.kthLargest(Util.moduleTimeList, 0, Util.moduleTimeList.length-1, k)
            println("threshold: " + threshold)
//            Util.moduleTimeList = null
            //clear moduleTimeList in each node
            models.mapPartitions { iter =>
              Util.moduleTimeList = new Array[Long](subModuleNumber*16*100)
              Iterator.empty
            }.count()
          }  
        }        
        iteration += 1
      }
      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logger.info(s"[Epoch $i/$epochNum] Epoch finished. Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      saveState(pm.getState(), i)
      test(module, i)
    }

    saveModel(module)
    saveState(pm.getState())
    module
  }
}