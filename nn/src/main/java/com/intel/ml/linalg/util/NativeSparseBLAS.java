package com.intel.ml.linalg.util;

public class NativeSparseBLAS {
  static {
    System.loadLibrary("nativeSparseBLAS"); // Load native library libnativeSparseBLAS.so at runtime
  }

  /***
   * Computes matrix - vector product of a sparse matrix
   * stored in the CSR format.
   *
   * @param transa  case 'N' or 'n':
   *                y := alpha * A * x + beta * y
   *                case 'T' or 't':
   *                y := alpha * A' * x + beta * y
   * @param numRowsA  Number of rows of the Matrix A.
   * @param numColumnsA  Number of columns of the Matrix A.
   * @param alpha  scalar.
   * @param matdescra  Array of six elements, specifies properties of the matrix used
   *                   for operation. Only first four array elements are used, their possible values
   *                   are given in Table “Possible Values of the Parameter matdescra
   *                   (https://software.intel.com/en-us/node/520801#TBL2-6)”
   * @param a  Array containing non-zero elements of the matrix A.
   * @param rowIndices  Array of length m+1, a[0] to a[m-1] represents the indices of the first non-zero
   *            elements in each column of matrix A; a[m] represent the index of the last non-zero
   *            element in row m.
   * @param columnIndices  Array containing the column indices for each non-zero element of the matrix A.
   * @param totalNonzeroElements  Specifies the number of non-zero element of the matrix A.
   * @param x  Array of vector x.
   * @param beta scalar.
   * @param y  Array of vector y.
   */
  public static native void dcoomv (char transa, int numRowsA, int numColumnsA, double alpha, String matdescra, double[] a,
                                    int[] rowIndices, int[] columnIndices, int totalNonzeroElements, double[] x, double beta,  double[] y);

  /***
   * Computes matrix - vector product of a sparse general
   * matrix stored in the COO format(3-array variation)
   * with zero-based indexing.
   * @param transa  case 'N' or 'n':
   *                y := A * x
   *                case 'T' or 't':
   *                y := A' * x
   * @param numRows  Number of rows of the Matrix A.
   * @param a  Array containing non-zero elements of the matrix A.
   * @param rowPointers  Array of length m+1, a[0] to a[m-1] represents the indices of the first non-zero
   *            elements in each column of matrix A; a[m] represent the index of the last non-zero
   *            element in row m.
   * @param columnIndices  Array containing the column indices for each non-zero element of the matrix A.
   * @param x  Array, size is m.
   * @param y  Output.
   */
  public static native void dcsrgemv (char transa, int numRows, double[] a, int[] rowPointers, int[] columnIndices, double[] x, double[] y);

  /***
   * Computes matrix - vector product of a sparse matrix
   * stored in the CSR format.
   *
   * @param transa  case 'N' or 'n':
   *                y := alpha * A * x + beta * y
   *                case 'T' or 't':
   *                y := alpha * A' * x + beta * y
   * @param numRowsA  Number of rows of the Matrix A.
   * @param numColumnsA  Number of columns of the Matrix A.
   * @param alpha  scalar.
   * @param matdescra  Array of six elements, specifies properties of the matrix used
   *                   for operation. Only first four array elements are used, their possible values
   *                   are given in Table “Possible Values of the Parameter matdescra
   *                   (https://software.intel.com/en-us/node/520801#TBL2-6)”
   * @param a  Array containing non-zero elements of the matrix A.
   * @param rowPointers  Array of length m+1, a[0] to a[m-1] represents the indices of the first non-zero
   *            elements in each column of matrix A; a[m] represent the index of the last non-zero
   *            element in row m.
   * @param columnIndices  Array containing the column indices for each non-zero element of the matrix A.
   * @param x  Array of vector x.
   * @param beta scalar.
   * @param y  Array of vector y.
   */
  public static native void dcsrmv (char transa, int numRowsA, int numColumnsA, double alpha, String matdescra, double[] a,
                                    int[] rowPointers, int[] columnIndices, double[] x, double beta,  double[] y);


  public static native void dcscmv (char transa, int numRowsA, int numColumnsA, double alpha, String matdescra, double[] a,
                                    int[] columnPointers, int[] rowIndices, double[] x, double beta,  double[] y);

  public static native void dcscmv (char transa, int numRowsA, int numColumnsA, double alpha, String matdescra, double[] a,
                                    int aOffset, int[] columnPointers,
                                    int[] rowIndices, int rowIndicesOffset,double[] x, double beta,  double[] y);
  /***
   * Computes matrix - matrix product of a sparse matrix stored in the CSR format.
   *
   * @param transa  case: 'N' or 'n':
   *                C := alpha*A*B + beta*C
   *                case: 'T' or 't':
   *                C := alpha*A'*B + beta*C
   * @param m  Number of rows of the matrix A.
   * @param n  Number of columns of the matrix C.
   * @param k  Number of columns of the matrix A.
   * @param alpha  scalar.
   * @param matdescra  Array of six elements, specifies properties of the matrix used
   *                   for operation. Only first four array elements are used, their possible values
   *                   are given in Table “Possible Values of the Parameter matdescra
   *                   (https://software.intel.com/en-us/node/520801#TBL2-6)”
   * @param a  Array containing non-zero elements of the matrix A.
   * @param rowPointers  Array of length m+1, a[0] to a[m-1] represents the indices of the first non-zero
   *            elements in each column of matrix A; a[m] represent the index of the last non-zero
   *            element in row m.
   * @param columnIndices  Array containing the column indices for each non-zero element of the matrix A.
   * @param b  size ldb by at least n for non-transposed matrix A and at least m for transposed
   *           for one-based indexing, and (at least k for non-transposedmatrix A and at least m
   *           for transposed, ldb ) for zero-based indexing.
   * @param ldb  Specifies the second dimension of b for zero-based indexing, as declared in the
   *             calling (sub)program.
   * @param beta  scalar.
   * @param c  Array, size ldc by n for one-based indexing, and (m, ldc) for zero-based indexing.
   * @param ldc  Specifies the leading dimension of c for one-based indexing, and the second
   *             dimension of c for zero-based indexing, as declared in the calling (sub)program.
   */
  public static native void dcsrmm(char transa, int m, int n, int k, double alpha, String matdescra, double[] a,
                                   int[] rowPointers, int[] columnIndices, double[] b, int ldb, double beta,
                                   double[] c, int ldc);

  public static native void dcscmm(char transa, int m, int n, int k, double alpha, String matdescra, double[] a,
                                   int[] columnPointers, int[] rowIndices, double[] b, int ldb, double beta,
                                   double[] c, int ldc);

  /***
   *   Adds a scalar multiple of compressed sparse vector to a full-storage vector.
   *
   *    y := a*x + y
   *
   * @param n  The number of elements in x and indices .
   * @param alpha  The scalar a.
   * @param x  Non-zero elements in x.
   * @param indices  The indices for the elements of x.
   * @param y  The full-storage vector y.
   */
  public static native void daxpyi(int n, double alpha, double[] x, int[] indices, double[] y);

  /***
   * Computes the dot product of a compressed sparse real vector by a full-storage real vector.
   *
   * @param n  The number of elements in x and indices .
   * @param x  Non-zero elements in x.
   * @param indices  The indices for the elements of x.
   * @param y  The full-storage vector y.
   * @return  x * y
   */
  public static native double ddoti(int n, double[] x, int[] indices, double[] y);


  /***/
  public static native void sparsespmm (int m, int n, int k, double[] a,
                                        int[] aColumnPointers, int[] aRowIndices, double[] g, int[] gColumnPointers, int[] gRowIndices,
                                        double[] v, int[] vColumnPointers, int[] vRowIndices);
}
