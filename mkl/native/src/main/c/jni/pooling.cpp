#include <jni.h>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

enum Algorithm { MAX, AVG, MIN };

template <typename DType>
class MKLPooling : public MKLLayer<DType>
{
 public:
  MKLPooling();
  ~MKLPooling();

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, size_t kernelHeight, size_t kernelWidth,
            size_t strideHeight, size_t strideWidth, int padHeight,
            int padWidth, int dimension, bool ceilMode, Algorithm pAl,
            const char *name);

  void updateOutput(DType *input, DType *output);
  void updateGradInput(DType *input, DType *gradOutput, DType *gradInput);

 private:
  std::shared_ptr<MKLData<DType>> workspace;

  size_t inputSize[4];
  size_t inputStrides[4];

  size_t kernelSize[2];

  size_t outputSizeCeil[4];
  size_t outputStridesCeil[4];

  size_t outputSizeFloor[4];
  size_t outputStridesFloor[4];

  size_t stride[2];
  int pad[2];

  // Algorithm for pooling : max, average, min. The default is MAX
  dnnAlgorithm_t algorithm;
  // When $mod(input + 2 * pad - kernel)$ is not eqal 0, the divisible will be
  // false.
  bool ceilMode;
};

template <typename DType>
MKLPooling<DType>::MKLPooling() : workspace(new MKLData<DType>)
{
}

template <typename DType>
MKLPooling<DType>::~MKLPooling()
{
}

template <typename DType>
void MKLPooling<DType>::init(size_t inputNumber, size_t inputChannel,
                             size_t inputHeight, size_t inputWidth,
                             size_t kernelHeight, size_t kernelWidth,
                             size_t strideHeight, size_t strideWidth,
                             int padHeight, int padWidth, int dimension,
                             bool ceilMode, Algorithm pAl, const char *name)
{
  MKLLayer<DType>::init(inputNumber, inputChannel, inputHeight, inputWidth,
                        dimension);

  this->name.assign(name);

  switch (pAl) {
    case MAX:
      algorithm = dnnAlgorithmPoolingMax;
      break;
    case AVG:
      algorithm = dnnAlgorithmPoolingAvg;
      break;
    case MIN:
      algorithm = dnnAlgorithmPoolingMin;
      break;
    default:
      algorithm = dnnAlgorithmPoolingMax;
  }

  stride[0] = strideWidth;
  stride[1] = strideHeight;

  kernelSize[0] = kernelWidth;
  kernelSize[1] = kernelHeight;

  pad[0] = -padWidth;
  pad[1] = -padHeight;

  this->ceilMode = ceilMode;

  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  inputStrides[0] = 1;
  for (int i        = 1; i < 4; i++)
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];

  // compute output
  outputSizeCeil[0] =
      computeOut(inputWidth, padWidth, kernelWidth, strideWidth, true);
  outputSizeCeil[1] =
      computeOut(inputHeight, padHeight, kernelHeight, strideHeight, true);
  outputSizeCeil[2] = this->inputSize[2];
  outputSizeCeil[3] = this->inputSize[3];

  outputSizeFloor[0] =
      computeOut(inputWidth, padWidth, kernelWidth, strideWidth, false);
  outputSizeFloor[1] =
      computeOut(inputHeight, padHeight, kernelHeight, strideHeight, false);
  outputSizeFloor[2] = this->inputSize[2];
  outputSizeFloor[3] = this->inputSize[3];

  // strides of input, kernel, output
  outputStridesFloor[0] = 1;
  outputStridesCeil[0]  = 1;
  for (int i = 1; i < 4; i++) {
    outputStridesFloor[i] = outputStridesFloor[i - 1] * outputSizeFloor[i - 1];
    outputStridesCeil[i]  = outputStridesCeil[i - 1] * outputSizeCeil[i - 1];
  }

  if (outputSizeCeil[0] == outputSizeFloor[0] &&
      outputSizeCeil[1] == outputSizeFloor[1])
    this->ceilMode = true;

  // create usr layout.
  this->input->createUsrLayout(dimension, inputSize, inputStrides);
  this->gradInput->createUsrLayout(dimension, inputSize, inputStrides);
  if (this->ceilMode) {
    this->output->createUsrLayout(dimension, outputSizeCeil, outputStridesCeil);
    this->gradOutput->createUsrLayout(dimension, outputSizeCeil,
                                      outputStridesCeil);
  } else {
    this->output->createUsrLayout(dimension, outputSizeFloor,
                                  outputStridesFloor);
    this->gradOutput->createUsrLayout(dimension, outputSizeFloor,
                                      outputStridesFloor);
  }

  /*
   * This is a trick that it must allocate memory for workspace.
   * Because defaultly, the sizeof workspace is <input size> * 2,
   * and so we set usrLayout defaultly to NULL.
   */
  // this->workspace->createUsrLayout(dimension, inputSize, inputStrides);
}

template <typename DType>
void MKLPooling<DType>::updateOutput(DType *input, DType *output)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  dnnError_t status  = E_UNIMPLEMENTED;
  dnnLayout_t layout = NULL;

// It's very stange, the address of input changes every time.
#ifdef DEBUG
  if (this->input->getUsrData() && this->input->getUsrData() != input)
    LOG(DBG) << "the address of input is not the same with preserved.";
#endif

  if (this->isFirstPass) {
    if (this->input->isUsePrev()) {
      layout = this->input->layoutPrev;
    }
    if (!layout) {
      status = dnnLayoutCreate<DType>(&layout, this->dimension, this->inputSize,
                                      this->inputStrides);
      CHECK_EQ(status, E_SUCCESS);
    }

    // forward
    status = dnnPoolingCreateForward<DType>(&(this->forwardPrim), NULL,
                                            algorithm, layout, kernelSize,
                                            stride, pad, dnnBorderZeros);
    CHECK_EQ(status, E_SUCCESS);
    this->input->createMklLayout(this->forwardPrim, dnnResourceSrc);
    this->output->createMklLayout(this->forwardPrim, dnnResourceDst);
    this->workspace->createMklLayout(this->forwardPrim, dnnResourceWorkspace);
    this->workspace->createConversion(true);

    // backward
    status = dnnPoolingCreateBackward<DType>(&(this->backwardPrim), NULL,
                                             algorithm, layout, kernelSize,
                                             stride, pad, dnnBorderZeros);
    CHECK_EQ(status, E_SUCCESS);

    // It's ok to set primitive as forwardPrim, because the relative type
    // is right.
    this->gradInput->createMklLayout(this->forwardPrim, dnnResourceSrc);
    this->gradOutput->createMklLayout(this->forwardPrim, dnnResourceDst);
    if (! this->input->isUsePrev()) {
      dnnLayoutDelete<DType>(layout);
    } else if (this->input->layoutPrev != layout) {
      // TODO We should add this code to other layers.
      dnnLayoutDelete<DType>(layout);
    }

    // the first pass we only create the layout, primitive, which are only
    // created the first time and not change.
    this->isFirstPass = false;
  }

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  this->input->setUsrData(input);
  this->input->createConversion();

  this->output->setUsrData(output);
  this->output->createConversion(!(ceilMode));
  this->workspace->setZero();
  // this->output->setZero();

  void *resources[dnnResourceNumber];
  resources[dnnResourceSrc]       = this->input->getConvertedData();
  resources[dnnResourceDst]       = this->output->getData();
  resources[dnnResourceWorkspace] = this->workspace->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->output->getUsrData()),
                   outputSizeCeil[3], outputSizeCeil[2], outputSizeCeil[1],
                   outputSizeCeil[0],
                   "Pooling forward output data generated by MKL2017");
#endif

  if (!this->output->isUseNext()) {
    if (ceilMode) {
      this->output->backToUsr();
    } else {
      this->output->cutLastRowColumn(outputStridesCeil, outputSizeFloor,
                                     outputStridesFloor);
    }
  }

#ifdef DEBUG
  printData<DType>(reinterpret_cast<DType *>(this->output->getUsrData()),
                   outputSizeFloor[3], outputSizeFloor[2], outputSizeFloor[1],
                   outputSizeCeil[0],
                   "Pooling forward output data generated by MKL2017");
#endif
}

template <typename DType>
void MKLPooling<DType>::updateGradInput(DType *input, DType *gradOutput,
                                        DType *gradInput)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

#ifdef DEBUG
  LOG(DBG) << "gradOutput = " << gradOutput
           << " dataUsr = " << this->gradOutput->getUsrData();
#endif

  // Because the address will change every time, so we need create conversion
  // every forward/backward.
  this->gradInput->setUsrData(gradInput);
  this->gradInput->createConversion();
  // Note: can't be deleted, because mkl dnn will not delete exist data
  this->gradInput->setZero();

  this->gradOutput->setUsrData(gradOutput);
  this->gradOutput->createConversion(!(ceilMode));
  // this->gradOutput->setZero();

  if (!ceilMode)
    this->gradOutput->padLastRowColumn(outputSizeFloor, outputStridesFloor,
                                       outputSizeCeil, outputStridesCeil);

  void *resources[dnnResourceNumber];
  resources[dnnResourceDiffDst]   = this->gradOutput->getConvertedData();
  resources[dnnResourceDiffSrc]   = this->gradInput->getData();
  resources[dnnResourceWorkspace] = this->workspace->getData();

  dnnError_t status;
  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  CHECK_EQ(status, E_SUCCESS);
  PERFEND("main computing");

  if (!this->gradInput->isUsePrev()) this->gradInput->backToUsr();
}

template <typename ArrayType, typename DType>
jlong JNIPoolingInit(JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel, jint inputHeight,
                     jint inputWidth, jint kernelHeight, jint kernelWidth,
                     jint strideHeight, jint strideWidth, jint padHeight,
                     jint padWidth, jint dimension, jint ceilMode, jint pAl,
                     jstring name)
{
  const char *jName = env->GetStringUTFChars(name, NULL);
  MKLPooling<DType> *pool = new MKLPooling<DType>();
  pool->init(inputNumber, inputChannel, inputHeight, inputWidth, kernelHeight,
             kernelWidth, strideHeight, strideWidth, padHeight, padWidth,
             dimension, ceilMode, static_cast<Algorithm>(pAl), jName);

  return reinterpret_cast<jlong>(pool);
}

template <typename ArrayType, typename DType>
void JNIPoolingUpdateOutput(JNIEnv *env, jclass thisClass, ArrayType input,
                            jint inputOffset, ArrayType output,
                            jint outputOffset, long classPtr)
{
  DType *jInputStart =
      reinterpret_cast<DType *>(env->GetPrimitiveArrayCritical(input, 0));
  DType *jOutputStart =
      reinterpret_cast<DType *>(env->GetPrimitiveArrayCritical(output, 0));

  DType *jInput  = jInputStart + inputOffset;
  DType *jOutput = jOutputStart + outputOffset;

  MKLPooling<DType> *ptr = reinterpret_cast<MKLPooling<DType> *>(classPtr);
  ptr->updateOutput(jInput, jOutput);

  env->ReleasePrimitiveArrayCritical(input, jInputStart, 0);
  env->ReleasePrimitiveArrayCritical(output, jOutputStart, 0);
}

template <typename ArrayType, typename DType>
void JNIPoolingUpdateGradInput(JNIEnv *env, jclass thisClass, ArrayType input,
                               jint inputOffset, ArrayType outputDiff,
                               jint outputDiffOffset, ArrayType inputDiff,
                               jint inputDiffOffset, long classPtr)
{
  DType *jInputStart =
      reinterpret_cast<DType *>(env->GetPrimitiveArrayCritical(input, 0));
  DType *jOutputDiffStart =
      reinterpret_cast<DType *>(env->GetPrimitiveArrayCritical(outputDiff, 0));
  DType *jInputDiffStart =
      reinterpret_cast<DType *>(env->GetPrimitiveArrayCritical(inputDiff, 0));

  DType *jInput      = jInputStart + inputOffset;
  DType *jOutputDiff = jOutputDiffStart + outputDiffOffset;
  DType *jInputDiff  = jInputDiffStart + inputDiffOffset;

  MKLPooling<DType> *ptr = reinterpret_cast<MKLPooling<DType> *>(classPtr);
  ptr->updateGradInput(jInput, jOutputDiff, jInputDiff);

  env->ReleasePrimitiveArrayCritical(input, jInputStart, 0);
  env->ReleasePrimitiveArrayCritical(outputDiff, jOutputDiffStart, 0);
  env->ReleasePrimitiveArrayCritical(inputDiff, jInputDiffStart, 0);
}

// Macro
#define PoolingInit(DType, JType, JArrayType)                                 \
  JNIEXPORT                                                                   \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_PoolingInit##DType(  \
      JNIEnv *env, jclass thisClass, jint inputNumber, jint inputChannel,     \
      jint inputHeight, jint inputWidth, jint kernelHeight, jint kernelWidth, \
      jint strideHeight, jint strideWidth, jint padHeight, jint padWidth,     \
      jint dimension, jint ceilMode, jint pAl, jstring name)                                \
  {                                                                           \
    return JNIPoolingInit<JArrayType, JType>(                                \
        env, thisClass, \
        inputNumber, inputChannel, inputHeight, inputWidth, kernelHeight,     \
        kernelWidth, strideHeight, strideWidth, padHeight, padWidth,          \
        dimension, ceilMode, pAl, name);                                            \
  }

#define PoolingForward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                    \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_PoolingForward##DType( \
      JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,       \
      JArrayType output, jint outputOffset, long classPtr)                     \
  {                                                                            \
    JNIPoolingUpdateOutput<JArrayType, JType>(                                 \
        env, thisClass, input, inputOffset, output, outputOffset, classPtr);   \
  }

#define PoolingBackward(DType, JType, JArrayType)                             \
  JNIEXPORT                                                                   \
  void JNICALL                                                                \
      Java_com_intel_analytics_sparkdl_mkl_MKL_PoolingBackward##DType(        \
          JNIEnv *env, jclass thisClass, JArrayType input, jint inputOffset,  \
          JArrayType outputDiff, jint outputDiffOffset, JArrayType inputDiff, \
          jint inputDiffOffset, long classPtr)                                \
  {                                                                           \
    JNIPoolingUpdateGradInput<JArrayType, JType>(                             \
        env, thisClass, input, inputOffset, outputDiff, outputDiffOffset,     \
        inputDiff, inputDiffOffset, classPtr);                                \
  }

#ifdef __cplusplus
extern "C" {
#endif

  // Double
  PoolingInit(Double, jdouble, jdoubleArray);
  PoolingForward(Double, jdouble, jdoubleArray);
  PoolingBackward(Double, jdouble, jdoubleArray);

  // Float
  PoolingInit(Float, jfloat, jfloatArray);
  PoolingForward(Float, jfloat, jfloatArray);
  PoolingBackward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
