package com.intel.analytics.bigdl.mkl;

/**
 * MKL DNN Library Wrapper for JVM
 */
public class MklDnnFloat {
    static {
        if (!MKL.isMKLLoaded()) {
            throw new RuntimeException("MKL is unloaded.");
        }
    }

    /**
     * create mkl layout, which is based on attributes user defined
     *
     * @param dimension tensor dimension
     * @param size      tensor size, for instance, 224 * 224 * 3 * 32.
     * @param strides   tensor strides
     * @return mkl layout pointer
     */
    public native static long layoutCreate(int dimension,
                                           long[] size,
                                           long[] strides);

    public native static void layoutDelete(long layout);

    /**
     * create mkl layout from primitive
     *
     * @param primitive layer primitive in mkl dnn
     * @param type      resource type
     * @return mkl layout pointer
     */
    public native static long layoutCreateFromPrimitive(long primitive,
                                                        int type);

    /**
     * return memory size / layout size from mkl layout pointer
     *
     * @param layout mkl layout pointer
     * @return memory size / layout size
     */
    public native static long layoutGetMemorySize(long layout);

    /**
     * compare two layouts and return the result.
     *
     * @param layout1
     * @param layout2
     * @return If layouts are the same, return 1.
     */
    public native static int layoutCompare(long layout1,
                                           long layout2);

    /**
     * call dnnAllocateBuffer_F32 to create align memroy.
     *
     * @param layout layoutPtr created by @layoutCreate
     * @return mem ptr allocated.
     */
    public native static long allocateBuffer(long layout);

    /**
     * delete the memory allocated by @allocateBuffer.
     *
     * @param memoryPtr
     */
    public native static void releaseBuffer(long memoryPtr);

    /**
     * copy from mkl buffer to another mkl buffer
     *
     * @param dst  destination
     * @param src  source
     * @param size the size (bytes) will be copied.
     */
    public native static void buffercpy(long dst, long src, long size);

    /**
     * set the buffer all zero, it is implemented by openmp for speeding up.
     *
     * @param buffer buffer allocated by dnnAllocateBuffer_F32
     * @param size size of buffer
     */
    public native static void setZero(long buffer, long size);

    /**
     * delete mkl primitive
     *
     * @param primitive primitive will be deleted.
     */
    public native static void deletePrimitive(long primitive);

    /**
     * create conversion primitive from layout1 to layout2
     *
     * @param layout1 src layout
     * @param layout2 dst layout
     * @return
     */
    public native static long conversionCreate(long layout1,
                                               long layout2);

    /**
     * convert mkl storage to an array. Be attention, the size should be the same.
     *
     * @param usr       user defined array
     * @param usrOffset the array offset
     * @param mkl       mkl storage
     * @param primitive mkl primitive
     */
    public native static void conversionExecuteToUsr(float[] usr,
                                                     int usrOffset,
                                                     long mkl,
                                                     long primitive);

    /**
     * convert the elements in an array to mkl storage.
     *
     * @param usr       user defined array
     * @param usrOffset the array offset
     * @param mkl       mkl storage
     * @param primitive mkl primitive
     */
    public native static void conversionExecuteToMkl(float[] usr,
                                                     int usrOffset,
                                                     long mkl,
                                                     long primitive);

    /**
     * convert mkl storage to another mkl storage
     *
     * @param src       src
     * @param dst       destination
     * @param primitive mkl primitive
     */
    public native static void conversionExecuteMklToMkl(long src, long dst, long primitive);

    /**
     * call dnnExecute_F32 t to compute
     * @param resources resources array, it contains all pointers of mkl storage
     * @param primitive mkl primitive
     * @return if success, return E_SUCCESS, otherwise return error number.
     */
    public native static int execute(long[] resources, long primitive);

    // convolution wrapper

    /**
     * create convolution primitive.
     *
     * @param algorithm  direct default
     * @param groups
     * @param dimension
     * @param inputSize
     * @param outputSize
     * @param weightSize
     * @param strides
     * @param pad        it must be nagtive.
     * @param boderType
     * @return mkl primitive
     */
    public native static long convolutionCreateForward(int algorithm,
                                                       long groups,
                                                       long dimension,
                                                       long[] inputSize,
                                                       long[] outputSize,
                                                       long[] weightSize,
                                                       long[] strides,
                                                       int[] pad,
                                                       int boderType);

    /**
     * create gradient input primitive
     *
     * @param algorithm
     * @param groups
     * @param dimension
     * @param inputSize
     * @param outputSize
     * @param weightSize
     * @param strides
     * @param pad
     * @param borderType
     * @return mkl primitive
     */
    public native static long convolutionCreateBackwardData(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] inputSize,
                                                            long[] outputSize,
                                                            long[] weightSize,
                                                            long[] strides,
                                                            int[] pad,
                                                            int borderType);

    /**
     * create gradient weight primitive
     * @param algorithm
     * @param groups
     * @param dimension
     * @param inputSize
     * @param outputSize
     * @param weightSize
     * @param strides
     * @param pad
     * @param borderType
     * @return mkl primitive
     */
    public native static long convolutionCreateBackwardKernel(int algorithm,
                                                              long groups,
                                                              long dimension,
                                                              long[] inputSize,
                                                              long[] outputSize,
                                                              long[] weightSize,
                                                              long[] strides,
                                                              int[] pad,
                                                              int borderType);

    /**
     * create gradient bias primitive
     * @param algorithm
     * @param groups
     * @param dimension
     * @param outputSize
     * @return mkl primitive
     */
    public native static long convolutionCreateBackwardBias(int algorithm,
                                                            long groups,
                                                            long dimension,
                                                            long[] outputSize);

    // ReLU
    public native static long reluCreateForward(long layout, float nagtiveSlope);

    public native static long reluCreateBackward(long layout1, long layout2, float nagtiveSlope);

    // Pooling
    public native static long poolCreateForward(int algorithm,
                                                long layout,
                                                long[] kernelSize,
                                                long[] stride,
                                                int[] pad,
                                                int borderType);

    public native static long poolCreateBackward(int algorithm,
                                                 long layout,
                                                 long[] kernelSize,
                                                 long[] stride,
                                                 int[] pad,
                                                 int borderType);

    // Linear
    public native static long linearCreateForwardBias(long dimension,
                                                      long[] inputSize,
                                                      long outputChannel);

    public native static long linearCreateBackData(long dimension,
                                                   long[] inputSize,
                                                   long outputChannel);

    public native static long linearCreateBackWeight(long dimension,
                                                     long[] inputSize,
                                                     long outputChannel);

    public native static long linearCreateBackBias(long dimension,
                                                   long[] outputSize);

    // LRN
    public native static long lrnCreateForward(long layout,
                                               long size,
                                               float alpha,
                                               float beta,
                                               float k);

    public native static long lrnCreateBackward(long layout1,
                                                long layout2,
                                                long size,
                                                float alpha,
                                                float beta,
                                                float k);

    public native static long concatCreate(long numConcats,
                                           long[] layouts);

    public native static long splitCreate(long numConcats,
                                          long layout,
                                          long[] splitsDistri);

    public native static long sumCreate(long numInputs,
                                        long layout,
                                        float[] coefficients);

    public native static long batchNormCreateForward(long layout,
                                                     float eps);

    public native static long batchNormCreateBackward(long layout,
                                                      float eps);

    public native static long batchNormCreateScaleShift(long layout,
                                                        float eps);

    public native static long setScaleShift(int affine,
                                            float[] weight,
                                            long weightOffset,
                                            float[] bias,
                                            long biasOffset,
                                            long scaleShift,
                                            int num);
    public native static long setGradScaleShift(int affine,
                                                float[] weight,
                                                long weightOffset,
                                                float[] bias,
                                                long biasOffset,
                                                long scaleShift,
                                                int num);

    public native static void unPadding(float[] to,
                                        long offset,
                                        long from,
                                        long fromStrides[],
                                        long toSize[],
                                        long toStrides[]);

    public native static void padding(float[] from,
                                      long offset,
                                      long to,
                                      long fromSize[],
                                      long fromStrides[],
                                      long toSize[],
                                      long toStrides[]);
}
