package com.intel.analytics.bigdl.mkl;

/**
 * MKL DNN Library Wrapper for JVM
 */
public class MklDnnFloat {
    public native static long layoutCreate(int dimension,
                                           long[] size,
                                           long[] strides);
    public native static long layoutCreateFromPrimitive(long primitive,
                                                        int type);

    public native static long layoutGetMemorySize(long layout);

    /**
     * compare two layouts and return the result.
     * @param layout1
     * @param layout2
     * @return If layouts are the same, return 1.
     */
    public native static int layoutCompare(long layout1,
                                           long layout2);

    public native static void deletePrimitive(long primitive);

    public native static long allocateBuffer();
    public native static void releaseBuffer();

    /**
     * create conversion primitive from layout1 to layout2
     * @param layout1 src layout
     * @param layout2 dst layout
     * @return
     */
    public native static long conversionCreate(long layout1,
                                               long layout2);
    public native static void conversionExecuteToUsr(float[] usr,
                                                     int usrOffset,
                                                     float[] mkl,
                                                     long primitive);
    public native static void conversionExecuteToMkl(float[] usr,
                                                     int usrOffset,
                                                     float[] mkl,
                                                     long primitive);

    // convolution wrapper
    public native static long convolutionCreateForward(int algorithm,
                                                       long groups,
                                                       long dimension,
                                                       long[] inputSize,
                                                       long[] outputSize,
                                                       long[] weightSize,
                                                       long[] strides,
                                                       int [] pad,
                                                       int boderType);
    public native static long convolutionCreateBackwardData(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] inputSize,
                                                            long[] outputSize,
                                                            long[] weightSize,
                                                            long[] strides,
                                                            int[] pad,
                                                            int borderType);
    public native static long convolutionCreateBackwardKernel(int algorithm,
                                                              long groups,
                                                              long dimension,
                                                              long[] inputSize,
                                                              long[] outputSize,
                                                              long[] weightSize,
                                                              long[] strides,
                                                              int[] pad,
                                                              int borderType);
    public native static long convolutionCreateBackwardBias(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] outputSize);
    public native static void convolutionForwardExecute(float[] input,
                                                        float[] weight,
                                                        float[] bias,
                                                        float[] output,
                                                        long primitive);
    public native static void convolutionBackwardDataExecute(float[] gradInput,
                                                             float[] gradOutput,
                                                             float[] backWeight,
                                                             long primitive);
    public native static void convolutionBackwardKernelExecute(float[] input,
                                                               float[] gradOutput,
                                                               float[] gradWeight,
                                                               long primitive);
    public native static void convolutionBackwardBiasExecute(float[] gradOutput,
                                                             float[] gradBias,
                                                             long primitive);
}
