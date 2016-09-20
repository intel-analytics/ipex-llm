#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT
void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_SetPrevFloat(
    JNIEnv *env, jclass thisClass, long prev, long curr)
{
  MKLLayer<float>::setPrev(prev, curr);
}

JNIEXPORT
void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_SetPrevDouble(
    JNIEnv *env, jclass thisClass, long prev, long curr)
{
  MKLLayer<double>::setPrev(prev, curr);
}

#ifdef __cplusplus
}
#endif
