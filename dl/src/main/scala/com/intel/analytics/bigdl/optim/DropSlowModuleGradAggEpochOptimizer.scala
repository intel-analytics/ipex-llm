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

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.nn.{Criterion, Module}
import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.parameters._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table, Util}
import org.apache.spark.{SparkEnv, TaskContext}

import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.reflect._
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object DropSlowModuleGradAggEpochOptimizer {
  val subModuleNumber = System.getProperty(
    "bigdl.optim.BetterGradAggEpochOptimizer.subModuleNumber",
    (Runtime.getRuntime().availableProcessors() / 2).toString()).toInt

  val dropModulePercentage = System.getProperty(
    "bigdl.optim.dropModulePercentage", "0.04").toDouble
  require(dropModulePercentage>=0 && dropModulePercentage<0.5)
  
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

  var weightSyncTime = 0L
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

  var ps = new AllReduceParameter[T]()
  private def init() = {
    val broadcast = dataSet.getSparkContext().broadcast((module, criterion))
    val _computeThresholdBatchsize = computeThresholdBatchsize
    val _partitionNum = dataSets.getPartitionNum()
    
    val models = dataSet.partitions().mapPartitions(i => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      
      val test = (0 until subModuleNumber).map { i =>
        val localModule = broadcastModule.cloneModule()
        val localCriterion = broadcastCriterion.cloneCriterion()
        val (weights, grads) = localModule.getParameters()        
        if(i == 0) CachedModel(localModule, localCriterion, weights, grads, T(),
          new Array[Long](subModuleNumber*_computeThresholdBatchsize))
        else CachedModel(localModule, localCriterion, weights, grads, T())}
      AllReduceParameter.taskSize = broadcastModule.getParameters()._1.nElement() / _partitionNum
      AllReduceParameter.extraSize = broadcastModule.getParameters()._1.nElement() % _partitionNum      
      AllReduceParameter.tlength = broadcastModule.getParameters()._1.nElement()  
      ps.init(broadcastModule.getParameters()._1)
      require(ps.parameterBuffer != null)
      Iterator(test.toArray)
    }).persist()
    models.setName("modelRDD")
    logger.info("Cache models...")
    models.count()
    logger.info("Cache models... done")
    models
  }

  private val computeThresholdBatchsize = 100
  val multiThreadModels = init()
  
  override def optimize(): Module[Tensor[T], Tensor[T], T] = {
    // don't send whole Optimizer in closure
    val broadcastEV = dataSets.getSparkContext().broadcast(ev)

    val sc = dataSets.getSparkContext()
    val _partitionNum = dataSets.getPartitionNum()
    var wallClockTime = 0L
    val epochNum = maxEpoch.getOrElse(20)    
    var timeout = Long.MaxValue
    var threshold = 0L    
    var iteration = 0
    val idealSubModulesNum = _partitionNum * subModuleNumber    
    val _moduleTimeIgnoredNum = 200
    val _computeThresholdBatchsize = 100
    var _moduleTimeList: Array[Long] = null

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

      
      
      while (!dataSets.epochFinished()) {
        val lossSum = sc.accumulator(0.0, "loss sum")
        val recordsNum = sc.accumulator(0, "record number")
        val stackCount = sc.accumulator(0, "stack count")        
        metrics.set("computing time for each node", mutable.ArrayBuffer[Double](), sc)        
        metrics.set("construct tensor time", 0.0, sc, _partitionNum)
        metrics.set("computing time average", 0.0, sc, _partitionNum)
        metrics.set("prepare time", 0.0, sc, _partitionNum)        
        metrics.set("aggregate gradient time", 0.0, sc, _partitionNum)
        metrics.set("task1 time from worker", mutable.ArrayBuffer[Double](), sc)
        metrics.set("task2 time from worker", mutable.ArrayBuffer[Double](), sc)
        metrics.set("worker sync weight average", 0.0, sc, _partitionNum)
        metrics.set("sync weight for each node", mutable.ArrayBuffer[Double](), sc)
        metrics.set("gradient sync average", 0.0, sc, _partitionNum)
        metrics.set("update weights average", 0.0, sc, _partitionNum)
        metrics.set("gradient sync for each node", mutable.ArrayBuffer[Double](), sc)
        metrics.set("update weights average", 0.0, sc, _partitionNum)
        
        val _metrics = metrics
        val _classTag = classTag[T]
        val start = System.nanoTime()
        val finishedModuleNum = dataSets.fetch().zipPartitions(models, multiThreadModels, true)(
          (data, modelIter, multiThreadModuleIter) => {
            val workStart = System.nanoTime()
            var tmp = System.nanoTime()
            val localMTCaches = multiThreadModuleIter.next()
            val localCaches = modelIter.next()
            val curPid = TaskContext.getPartitionId()
            
            val syWStart = System.nanoTime()
            val getWeightsTasks = ps.getWeights(localCaches.weight, _partitionNum)

            val initGTasks = (0 until subModuleNumber).map(i => pool.submit(new Runnable {
             override def run() = {
               localMTCaches(i).model.training()
               localMTCaches(i).model.zeroGradParameters()
             }
          }))
            getWeightsTasks.foreach(_.get())
            _metrics.add("worker sync weight average", System.nanoTime()-syWStart)
            _metrics.add("sync weight for each node", System.nanoTime()-syWStart)
            
            val weightCopyTasks = (0 until subModuleNumber).map(i => pool.submit(new Runnable {
              override def run(): Unit = {
                localMTCaches(i).weight.copy(localCaches.weight)  
              }
            }))
            
            val localEV = broadcastEV.value
            tmp = System.nanoTime()
            val tensorBuffer = new Array[(Tensor[T], Tensor[T])](subModuleNumber)
            
              val batch = data.next()
              var b = 0
              while(b < subModuleNumber) {
                tensorBuffer(b) = batch.next()
                b += 1
              }
            _metrics.add("construct tensor time", System.nanoTime() - tmp)
            initGTasks.foreach(_.get())
            weightCopyTasks.foreach(_.get())
            _metrics.add("prepare time", System.nanoTime() - tmp)

            // ======================Start train models===================================
            tmp = System.nanoTime()
            val moduleTimeList = localMTCaches(0).moduleTimeList
            val pre = (iteration%_computeThresholdBatchsize)*subModuleNumber
            val computeThreads = (0 until subModuleNumber).map(i => new Callable[Int] {
              def call(): Int = {
                try {
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
                  moduleTimeList(i + pre) = System.nanoTime() - start + weightSyncTime
                  i
                }
//                catch {
//                  case e: Exception => e.printStackTrace()
//                    -1
//                }
              }
            })

            if(iteration > _moduleTimeIgnoredNum+_computeThresholdBatchsize-1) {
              timeout = ((threshold-weightSyncTime)/1e6).toLong
            }
            val threads = pool.invokeAll(computeThreads.asJava, timeout, TimeUnit.MILLISECONDS)

            val computingTime = System.nanoTime() - tmp
            _metrics.add("computing time average", computingTime)
            _metrics.add("computing time for each node", computingTime)

            val finishedThreads = threads.asScala.filter(!_.isCancelled)
              .map(_.get()).filter(_ != -1)            

            stackCount += finishedThreads.size
            finishedThreads.foreach{index =>
              lossSum += lossArray(index)
              recordsNum += recordsArray(index)
            }

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
            } else {
              localCaches.model.zeroGradParameters()
            }
            _metrics.add("aggregate gradient time", System.nanoTime() - tmp)
            require(localCaches.gradient != null)
            require(ps.parameterBuffer != null)
            ps.putGradients(localCaches.gradient,
              curPid, _partitionNum)
            _metrics.add("task1 time from worker", System.nanoTime() - workStart)
            Iterator(finishedThreads.size)
          }).reduce(_ + _)
        metrics.set("task1 time from driver", System.nanoTime() - start)
        
        if(finishedModuleNum >= idealSubModulesNum * 0.5) {          
          val finishedModuleNumGType = ev.fromType[Int](finishedModuleNum)
          val value = ev.fromType(lossSum.value / stackCount.value)
          val _optm = optm
          val _config = config
          
          val task2Start = System.nanoTime()
          models.mapPartitions (modelIter => {
            val task2WorkerStart = System.nanoTime()
            val localModel = modelIter.next
            val curPid = TaskContext.getPartitionId()
            var tmp = System.nanoTime()
            val params = new Array[CompressedTensor[T]](_partitionNum)
            val getGradients = ps.getGradients(params, curPid, _partitionNum)
            getGradients.foreach(_.get())
            _metrics.add("gradient sync average", System.nanoTime()-tmp)
            _metrics.add("gradient sync for each node", System.nanoTime()-tmp) 
              
            params.head.deCompress(ps.gradients)

            tmp = System.nanoTime()
            _optm.optimize(_ => (value, ps.gradients),
              ps.weights, _config, ps.state)
            ps.putWeights(curPid)
            _metrics.add("task2 time from worker", System.nanoTime()-task2WorkerStart)
            Iterator.empty
          }).count()
          metrics.set("task2 time from driver", System.nanoTime()-task2Start)
          
          accumulateCount += recordsNum.value
          val end = System.nanoTime()
          logger.info(s"[Epoch $i/$epochNum $accumulateCount/${dataSets.total()}] Train ${
            recordsNum.value
          } in ${(end - start) / 1e9}seconds. " +
            s"Throughput is ${recordsNum.value / ((end - start) / 1e9)} records/second. Loss is ${
              lossSum.value / stackCount.value
            }. ")
          logger.info("\n" + metrics.summary())

          // compute threshold
          if(iteration%_computeThresholdBatchsize==_computeThresholdBatchsize-1 &&
            iteration>=(_computeThresholdBatchsize+_moduleTimeIgnoredNum-1)) {
            _moduleTimeList = multiThreadModels.mapPartitions{ iter =>
              iter.next().apply(0).moduleTimeList.iterator
            }.collect()
            val k = (dropModulePercentage*_computeThresholdBatchsize*idealSubModulesNum).toInt
            threshold = Util.kthLargest(_moduleTimeList, 0, _moduleTimeList.length-1, k)
            logger.info("threshold: " + threshold)

            // clear moduleTimeList in each node
            multiThreadModels.mapPartitions { iter =>
              iter.next.apply(0).moduleTimeList = new Array[Long](subModuleNumber*_computeThresholdBatchsize)
              Iterator.empty
            }.count()
          }
          iteration += 1
        } else {
          logger.info(s"Warning!!! Ignore this iteration as more than half " +
            s"module is dropped!! Finished modules number: ${finishedModuleNum}")
        }
      }
        
      val epochEnd = System.nanoTime()
      wallClockTime = wallClockTime + epochEnd - epochStart
      logger.info(s"[Epoch $i/$epochNum] Epoch finished. " +
        s"Wall clock time is ${wallClockTime / 1e6}ms")
      saveModel(module, i)
      saveState(pm.getState(), i)
      test(module, i)
    }

    saveModel(module)
    saveState(pm.getState())
    module
  }
}
