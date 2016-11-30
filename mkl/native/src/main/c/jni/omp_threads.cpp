#include <jni.h>
#include <omp.h>
#include <mkl.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    setNumThreads
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_setNumThreads
  (JNIEnv * env, jclass cls, jint num_threads) {
  omp_set_num_threads(num_threads);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    getNumThreads
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_getNumThreads
  (JNIEnv * env, jclass cls) {
  return omp_get_max_threads();
}
/*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsAdd
  * Signature: (I[FI[FI[FI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsAdd
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b,
   jint bOffset, jfloatArray y, jint yOffset) {
   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_b = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));
   vsAdd( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);
   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdAdd
  * Signature: (I[DI[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdAdd
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray b,
   jint bOffset, jdoubleArray y, jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_b = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdAdd( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsSub
  * Signature: (I[FI[FI[FI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsSub
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b,
   jint bOffset, jfloatArray y, jint yOffset) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_b = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vsSub( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdSub
  * Signature: (I[DI[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdSub
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray b,
   jint bOffset, jdoubleArray y, jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_b = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdSub( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

/*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsMul
  * Signature: (I[FI[FI[FI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsMul
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b,
   jint bOffset, jfloatArray y, jint yOffset) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_b = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vsMul( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdMul
  * Signature: (I[DI[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdMul
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray b,
   jint bOffset, jdoubleArray y, jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_b = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdMul( n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    vsDiv
 * Signature: (I[FI[FI[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsDiv
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b, jint bOffset,
  jfloatArray y, jint yOffset) {


   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_b = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vsDiv(n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    vdDiv
 * Signature: (I[DI[DI[DI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdDiv
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray b, jint bOffset,
  jfloatArray y, jint yOffset) {


   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_b = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdDiv(n, jni_a + aOffset, jni_b + bOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   }

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    vsPowx
 * Signature: (I[FIF[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsPowx
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloat b, jfloatArray y,
  jint yOffset) {

 jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
 jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

 vsPowx( n, jni_a + aOffset, b, jni_y + yOffset);

 env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
 env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdPowx
  * Signature: (I[DID[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdPowx
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdouble b, jdoubleArray y,
   jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdPowx( n, jni_a + aOffset, b, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    vsLn
 * Signature: (I[FI[FI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsLn
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
  jint yOffset) {

 jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
 jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

 vsLn( n, jni_a + aOffset, jni_y + yOffset);

 env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
 env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdLn
  * Signature: (I[DI[DI)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdLn
   (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
   jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vdLn( n, jni_a + aOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsExp
  * Signature: (I[FI[FI)V
  */
 JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsExp
   (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
   jint yOffset) {

  jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
  jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

  vsExp( n, jni_a + aOffset, jni_y + yOffset);

  env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
  env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

  /*
   * Class:     com_intel_analytics_bigdl_mkl_MKL
   * Method:    vdExp
   * Signature: (I[DI[DI)V
   */
 JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdExp
    (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
    jint yOffset) {

    jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
    jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

    vdExp( n, jni_a + aOffset, jni_y + yOffset);

    env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
    env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
  }

  /*
   * Class:     com_intel_analytics_bigdl_mkl_MKL
   * Method:    vsSqrt
   * Signature: (I[FI[FI)V
   */
  JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsSqrt
    (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
    jint yOffset) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vsSqrt( n, jni_a + aOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
  }

   /*
    * Class:     com_intel_analytics_bigdl_mkl_MKL
    * Method:    vdSqrt
    * Signature: (I[DI[DI)V
    */
  JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdSqrt
     (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
     jint yOffset) {

     jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
     jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

     vdSqrt( n, jni_a + aOffset, jni_y + yOffset);

     env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
     env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   }

   /*
    * Class:     com_intel_analytics_bigdl_mkl_MKL
    * Method:    vsLog1p
    * Signature: (I[FI[FI)V
    */
   JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsLog1p
     (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset, jfloatArray y,
     jint yOffset) {

    jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
    jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

    vsLog1p( n, jni_a + aOffset, jni_y + yOffset);

    env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
    env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   }

    /*
     * Class:     com_intel_analytics_bigdl_mkl_MKL
     * Method:    vdLog1p
     * Signature: (I[DI[DI)V
     */
   JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdLog1p
      (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
      jint yOffset) {

      jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
      jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

      vdLog1p( n, jni_a + aOffset, jni_y + yOffset);

      env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
      env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
    }

/*
    * Class:     com_intel_analytics_bigdl_mkl_MKL
    * Method:    vsLog1p
    * Signature: (I[FI[FI)V
    */
   JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdAbs
     (JNIEnv * env, jclass cls, jint n, jdoubleArray a, jint aOffset, jdoubleArray y,
     jint yOffset) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

    vdAbs( n, jni_a + aOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
 }

/*
 * Class:     com_intel_analytics_bigdl_mkl_MKL
 * Method:    vdDiv
 * Signature: (I[DI[DI[DI)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsAbs
  (JNIEnv * env, jclass cls, jint n, jfloatArray a, jint aOffset,
  jfloatArray y, jint yOffset) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   vsAbs(n, jni_a + aOffset, jni_y + yOffset);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
}

   /*
    * Class:     com_intel_analytics_bigdl_mkl_MKL
    * Method:    vsgemm
    * Signature: (CCIIIF[FII[FIIF[FII)V
    */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsgemm
  (JNIEnv * env, jclass cls, jchar transa, jchar transb, jint m, jint n,
   jint k, jfloat alpha, jfloatArray a, jint aOffset, jint lda, jfloatArray b,
   jint bOffset, jint ldb, jfloat beta, jfloatArray c, jint cOffset, jint ldc) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_b = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
   jfloat * jni_c = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(c, JNI_FALSE));

   CBLAS_TRANSPOSE jni_transa,jni_transb;
   if(transa == 't' || transa == 'T') jni_transa = CblasTrans; else jni_transa = CblasNoTrans;
   if(transb == 't' || transb == 'T') jni_transb = CblasTrans; else jni_transb = CblasNoTrans;

   cblas_sgemm(CblasColMajor, jni_transa, jni_transb, m, n, k, alpha, jni_a + aOffset, lda,
   jni_b + bOffset, ldb, beta, jni_c + cOffset, ldc);

   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
   env->ReleasePrimitiveArrayCritical(c, jni_c, 0);
}

    /*
     * Class:     com_intel_analytics_bigdl_mkl_MKL
     * Method:    vdgemm
     * Signature: (CCIIID[DII[DIID[DII)V
     */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdgemm
  (JNIEnv * env, jclass cls, jchar transa, jchar transb, jint m, jint n,
  jint k, jdouble alpha, jdoubleArray a, jint aOffset, jint lda, jdoubleArray b,
  jint bOffset, jint ldb, jdouble beta, jdoubleArray c, jint cOffset, jint ldc) {

  jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
  jdouble * jni_b = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(b, JNI_FALSE));
  jdouble * jni_c = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(c, JNI_FALSE));

  CBLAS_TRANSPOSE jni_transa,jni_transb;
  if(transa == 't' || transa == 'T') jni_transa = CblasTrans; else jni_transa = CblasNoTrans;
  if(transb == 't' || transb == 'T') jni_transb = CblasTrans; else jni_transb = CblasNoTrans;
  cblas_dgemm(CblasColMajor, jni_transa, jni_transb, m, n, k, alpha, jni_a + aOffset, lda,
  jni_b + bOffset, ldb, beta, jni_c + cOffset, ldc);

  env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
  env->ReleasePrimitiveArrayCritical(b, jni_b, 0);
  env->ReleasePrimitiveArrayCritical(c, jni_c, 0);
}

   /*
    * Class:     com_intel_analytics_bigdl_mkl_MKL
    * Method:    vsgemv
    * Signature: (SIIF[FII[FIIF[FII)V
    */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsgemv
  (JNIEnv * env, jclass cls, jchar trans, jint m, jint n,
   jfloat alpha, jfloatArray a, jint aOffset, jint lda, jfloatArray x,
   jint xOffset, jint incx, jfloat beta, jfloatArray y, jint yOffset, jint incy) {

   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jfloat * jni_x = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   CBLAS_TRANSPOSE jni_trans;
   if(trans == 't' || trans == 'T') jni_trans = CblasTrans; else jni_trans = CblasNoTrans;
   cblas_sgemv(CblasColMajor, jni_trans, m, n, alpha, jni_a + aOffset, lda, jni_x + xOffset, incx,
     beta, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
}

    /*
     * Class:     com_intel_analytics_bigdl_mkl_MKL
     * Method:    vdgemv
     * Signature: (SSIIID[DII[DIID[DII)V
     */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdgemv
   (JNIEnv * env, jclass cls, jchar trans, jint m, jint n,
   jdouble alpha, jdoubleArray a, jint aOffset, jint lda, jdoubleArray x,
   jint xOffset, jint incx, jdouble beta, jdoubleArray y, jint yOffset, jint incy) {

   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));
   jdouble * jni_x = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   CBLAS_TRANSPOSE jni_trans;
   if(trans == 't' || trans == 'T') jni_trans = CblasTrans; else jni_trans = CblasNoTrans;
   cblas_dgemv(CblasColMajor, jni_trans, m, n, alpha, jni_a + aOffset, lda, jni_x + xOffset, incx,
      beta, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsaxpy
  * Signature: (II[FII[FII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsaxpy
   (JNIEnv * env, jclass cls, jint n, jfloat a, jfloatArray x, jint xOffset, jint incx,
   jfloatArray y, jint yOffset, jint incy) {
   jfloat * jni_x = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   cblas_saxpy(n, a, jni_x + xOffset, incx, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdaxpy
  * Signature: (II[DII[DII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdaxpy
   (JNIEnv * env, jclass cls, jint n, jdouble a, jdoubleArray x, jint xOffset, jint incx,
   jdoubleArray y, jint yOffset, jint incy) {
   jdouble * jni_x = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   cblas_daxpy(n, a, jni_x + xOffset, incx, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsdot
  * Signature: (I[FII[FII)V
  */
JNIEXPORT float JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsdot
   (JNIEnv * env, jclass cls, jint n, jfloatArray x, jint xOffset, jint incx,
   jfloatArray y, jint yOffset, jint incy) {
   jfloat * jni_x = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   float res = cblas_sdot(n, jni_x + xOffset, incx, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
   return res;
 }

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vddot
  * Signature: (I[DII[DII)V
  */
JNIEXPORT double JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vddot
   (JNIEnv * env, jclass cls, jint n, jdoubleArray x, jint xOffset, jint incx,
   jdoubleArray y, jint yOffset, jint incy) {

   jdouble * jni_x = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));

   double res = cblas_ddot(n, jni_x + xOffset, incx, jni_y + yOffset, incy);

   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
   return res;
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsger
  * Signature: (I[FII[FII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsger
   (JNIEnv * env, jclass cls, jint m, jint n, jfloat alpha, jfloatArray x, jint xOffset,
   jint incx, jfloatArray y, jint yOffset, jint incy, jfloatArray a, jint aOffset, jint lda) {
   jfloat * jni_x = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jfloat * jni_y = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));
   jfloat * jni_a = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));

   cblas_sger(CblasColMajor, m, n, alpha, jni_x + xOffset, incx, jni_y + yOffset, incy, jni_a + aOffset, lda);

   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdger
  * Signature: (I[DII[DII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdger
   (JNIEnv * env, jclass cls, jint m, jint n, jdouble alpha, jdoubleArray x, jint xOffset,
   jint incx, jdoubleArray y, jint yOffset, jint incy, jdoubleArray a, jint aOffset, jint lda) {

   jdouble * jni_x = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));
   jdouble * jni_y = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(y, JNI_FALSE));
   jdouble * jni_a = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(a, JNI_FALSE));

   cblas_dger(CblasColMajor, m, n, alpha, jni_x + xOffset, incx, jni_y + yOffset, incy, jni_a + aOffset, lda);

   env->ReleasePrimitiveArrayCritical(a, jni_a, 0);
   env->ReleasePrimitiveArrayCritical(y, jni_y, 0);
   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vsscal
  * Signature: (II[FII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vsscal
   (JNIEnv * env, jclass cls, jint n, jfloat a, jfloatArray x, jint xOffset, jint incx) {
   jfloat * jni_x = reinterpret_cast<jfloat*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));

   cblas_sscal(n, a, jni_x + xOffset, incx);

   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

 /*
  * Class:     com_intel_analytics_bigdl_mkl_MKL
  * Method:    vdscal
  * Signature: (II[DII)V
  */
JNIEXPORT void JNICALL Java_com_intel_analytics_bigdl_mkl_MKL_vdscal
   (JNIEnv * env, jclass cls, jint n, jdouble a, jdoubleArray x, jint xOffset, jint incx) {

   jdouble * jni_x = reinterpret_cast<jdouble*>(env->GetPrimitiveArrayCritical(x, JNI_FALSE));

   cblas_dscal(n, a, jni_x + xOffset, incx);

   env->ReleasePrimitiveArrayCritical(x, jni_x, 0);
}

#ifdef __cplusplus
}
#endif
