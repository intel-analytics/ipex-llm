package com.intel.analytics.bigdl.mkl;

public class MklDnnDouble {
    public native static long layoutCreate();
    public native static long layoutCreateFromPrimitive();

    public native static long layoutGetMemorySize();
    public native static int layoutCompare(long layout1, long layout2);

    public native static long allocateBuffer();
    public native static void releaseBuffer();

    public native static long conversionCreate();
    public native static void conversionExecute(float[] usr, float[] mkl,
                                                long primitive, int toMkl);

    // convolution wrapper
    public native static long convolutionCreateForward();
    public native static long convolutionCreateBackwardData();
    public native static long convolutionCreateBackwardKernel();
    public native static long convolutionCreateBackwardBias();
    public native static void convolutionForwardExecute();
    public native static void convolutionBackwardExecute();
}
