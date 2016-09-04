package com.intel.analytics.dllib.lib.tensor

import com.github.fommil.netlib.{F2jBLAS, NativeSystemBLAS}
import com.intel.ml.linalg.util.NativeSparseBLAS
import com.github.fommil.netlib.BLAS.getInstance

object SparseTensorBLAS {
  private val sparseblas = (System.getProperty("BLAS") == "NativeSystemBLAS")

  private val blas =
    if(System.getProperty("BLAS") == "NativeSystemBLAS") {
      new NativeSystemBLAS
    } else if(System.getProperty("BLAS") == "F2JBLAS") {
      new F2jBLAS
    } else {
      getInstance
    }

  def coodgemv(
                    alpha: Double,
                    A: SparseTensor[Double],
                    x: DenseTensor[Double],
                    beta: Double,
                    y: DenseTensor[Double]): Unit = {
    val xValues = x.storage().array()
    val yValues = y.storage().array()
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if(sparseblas) {
//      println(s"omp thread${System.getenv("OMP_NUM_THREADS")}")
      if(A.indices_order(0) == 0 && A.indices_order(1) == 1) {
        NativeSparseBLAS.dcoomv('N', mA, nA, alpha, "GLNF", Avals, Arows, Acols, Avals.length, xValues, beta, yValues)
      } else {
        NativeSparseBLAS.dcoomv('T', mA, nA, alpha, "GLNF", Avals, Acols, Arows, Avals.length, xValues, beta, yValues)
      }
    }else {
      coodgemv(alpha, Avals, Arows, Acols, xValues, beta, yValues)
    }
  }

  def coodgemv(alpha: Double, Avals: Array[Double], Arows: Array[Int], Acols: Array[Int], xValues: Array[Double], beta : Double,yValues: Array[Double]): Unit = {
    if (beta != 1) {
      blas.dscal(yValues.size, beta, yValues, 1)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow-1) += Aval * alpha * xValues(Acol-1)
      valueCounter += 1
    }
  }

  def coosgemv(
                        alpha: Float,
                        A: SparseTensor[Float],
                        x: DenseTensor[Float],
                        beta: Float,
                        y: DenseTensor[Float]): Unit = {
    val xValues = x.storage().array()
    val yValues = y.storage().array()
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = A._indices(A.indices_order(0))
    val Acols = A._indices(A.indices_order(1))

    if(beta == 0) {
      y.fill(0)
    } else if(beta != 1) {
      y.apply1(v => beta * v)
    }
    // Perform matrix-vector multiplication and add to y
    var valueCounter = 0
    while (valueCounter < Avals.length) {
      val Arow = Arows(valueCounter)
      val Acol = Acols(valueCounter)
      val Aval = Avals(valueCounter)
      yValues(Arow-1) += Aval * alpha * xValues(Acol-1)
      valueCounter += 1
    }

  }

  def csrdgemv(
                alpha: Double,
                A: SparseTensorCsr[Double],
                x: DenseTensor[Double],
                beta: Double,
                y: DenseTensor[Double]): Unit = {
    val xValues = x.storage().array()
    val yValues = y.storage().array()
    val mA: Int = A._shape(0)
    val nA: Int = A._shape(1)

    val Avals = A._values.array()
    val Arows = if(A.transposed) A._columns else A._rowIndices
    val Acols = if(A.transposed) A._rowIndices else A._columns

    if(sparseblas) {
      //      println(s"omp thread${System.getenv("OMP_NUM_THREADS")}")
      if(!A.transposed) {
        NativeSparseBLAS.dcsrmv('N', mA, nA, alpha, "GLNF", Avals, Arows, Acols, xValues, beta, yValues)
      } else {
        NativeSparseBLAS.dcsrmv('T', mA, nA, alpha, "GLNF", Avals, Acols, Arows, xValues, beta, yValues)
      }
    }else {
      csrdgemv(A.transposed, mA, nA, alpha, Avals, A._rowIndices, A._columns, xValues, beta, yValues)
      throw new UnsupportedOperationException("unimplement java version")
    }
  }

  /** *
    * y := alpha * A * x + beta * y
    *
    * @param transposed  if matrix A is transposed
    * @param mA  number of rows in matrix A
    * @param nA  number of column in matrix A
    * @param alpha
    * @param Avals  Value Array of matrix A
    * @param Arows  rowIndices, Arows.length = if(!transposed) mA + 1 else nA + 1
    * @param Acols  columnIndices, Acols.length = Avals.length
    * @param xValues  Value array of vector x
    * @param beta
    * @param yValues  Value array of vector y
    */
  def csrdgemv(
                transposed: Boolean,
                mA: Int,
                nA: Int,
                alpha: Double,
                Avals: Array[Double],
                Arows: Array[Int],
                Acols: Array[Int],
                xValues: Array[Double],
                beta : Double,
                yValues: Array[Double]
                ): Unit = {
    // Perform matrix-vector multiplication and add to y
    if (!transposed) {
      var rowCounter = 0
      while (rowCounter < mA) {
        var i = Arows(rowCounter)
        val indEnd = Arows(rowCounter + 1)
        var sum = 0.0
        while (i < indEnd) {
          sum += Avals(i) * xValues(Acols(i))
          i += 1
        }
        yValues(rowCounter) = beta * yValues(rowCounter) + sum * alpha
        rowCounter += 1
      }
    } else {
      if (beta != 1) {
        blas.dscal(yValues.size, beta, yValues, 1)
      }
      // Perform matrix-vector multiplication and add to y
      var colCounterForA = 0
      while (colCounterForA < nA) {
        var i = Arows(colCounterForA)
        val indEnd = Arows(colCounterForA + 1)
        val xVal = xValues(colCounterForA) * alpha
        while (i < indEnd) {
          val rowIndex = Acols(i)
          yValues(rowIndex) += Avals(i) * xVal
          i += 1
        }
        colCounterForA += 1
      }
    }
  }
}
