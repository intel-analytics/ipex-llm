// scalastyle:off println
package com.intel.analytics.bigdl.ppml

import scala.math.random

/** Computes an approximation to pi */
object ScalaPi {
    def randPoint(): Int = {
        val x = random * 2 - 1
        val y = random * 2 - 1
        if (x*x + y*y <= 1) 1 else 0
    }
    def main(args: Array[String]): Unit = {
        val slices = if (args.length > 0) args(0).toInt else 2
        val n = args(0).toInt
        var count=0
        for (i <- 1 to n){
            count=count+randPoint()
        }
        println(s"Pi is roughly ${4.0 * count / (n - 1)}")
    }
}
// scalastyle:on println
