#include <stdio.h>
#include <vector>

#include "debug.h"
#include "layer.h"
#include "memory.h"
#include "utils.h"

using namespace std;

template <typename DType>
class MKLConcat : public MKLLayer<DType>
{
 public:
  MKLConcat();
  ~MKLConcat();

  void init(int numConcats, int dimension, int *size);

  void updateOutput(DType **input, DType *output);
  void updateGradInput(DType **gradInput, DType *gradOutput);

  // attention, we will override the four variables of MKLLayer
  vector<shared_ptr<MKLData<DType>>> input;
  vector<shared_ptr<MKLData<DType>>> gradInput;

 private:
  // this method is not the same as createMklLayout in MKLMemory
  void firstPass();
  void preExecute(DType *input);

  int numConcats;  // number of concats
  size_t *numSplits;
};

template <typename DType>
MKLConcat<DType>::MKLConcat() : numSplits(NULL), numConcats(0)
{
  // TODO
}

template <typename DType>
MKLConcat<DType>::~MKLConcat()
{
  // TODO
  delete[] numSplits;
}

template <typename DType>
void MKLConcat<DType>::init(int numConcats, int dimension, int *size)
{
  this->numConcats = numConcats;
  this->dimension  = dimension;
  this->numSplits  = new size_t[numConcats];

  size_t inputSize[dimension];
  size_t inputStrides[dimension];
  size_t outputSize[dimension];
  size_t outputStrides[dimension];

  int offset      = 0;
  size_t channels = 0;

  for (int i = 0; i < numConcats; i++) {
    input.push_back(shared_ptr<MKLData<DType>>(new MKLData<DType>));
    gradInput.push_back(shared_ptr<MKLData<DType>>(new MKLData<DType>));

    // set the size.
    // the size of every channel should be gaved in size.
    // the dimension of every channel should be the same.
    inputStrides[0] = 1;
    inputSize[0]    = size[offset];
    for (int j = 1; j < dimension; j++) {
      inputSize[j]    = size[offset + j];
      inputStrides[j] = inputStrides[j - 1] * inputSize[j - 1];
    }
    offset += dimension;

    // we must be sure that inputSize[2] is channels, or it will be 1
    // if dimension == 2, which means there are only height and width. -> height
    // if dimension >  2, which means there is channel in the tensor, -> channel
    numSplits[i] = dimension <= 2 ? inputSize[1] : inputSize[2];
    channels += numSplits[i];

    this->input[i]->createUsrLayout(dimension, inputSize, inputStrides);
    this->gradInput[i]->createUsrLayout(dimension, inputSize, inputStrides);
  }

  // the output size should be equal to the first input size, besides channel
  // the channel of output (outputSize[2]) should be the sum of all
  // input channels.
  // the number of output is only 1
  outputStrides[0] = 1;
  outputSize[0]    = inputSize[0];
  for (int i = 1; i < dimension; i++) {
    if (i == 2)
      outputSize[i] = channels;
    else
      outputSize[i]  = inputSize[i];
    outputStrides[i] = outputStrides[i - 1] * outputSize[i - 1];
  }

  this->output->createUsrLayout(dimension, outputSize, outputStrides);
  this->gradOutput->createUsrLayout(dimension, outputSize, outputStrides);
}

template <typename DType>
void MKLConcat<DType>::firstPass()
{
  dnnLayout_t *layouts = new dnnLayout_t[numConcats];

  for (int i = 0; i < numConcats; i++) {
    layouts[i] = this->input[i]->getUsrLayout();
  }

  dnnError_t status = E_UNIMPLEMENTED;
  status =
      dnnConcatCreate<DType>(&(this->forwardPrim), NULL, numConcats, layouts);
  CHECK_EQ(status, E_SUCCESS);

  this->output->createMklLayout(this->forwardPrim, dnnResourceDst);
  this->gradOutput->createMklLayout(this->forwardPrim, dnnResourceDst);

  // backward
  status = dnnSplitCreate<DType>(&(this->backwardPrim), NULL, numConcats,
                                 this->gradOutput->getMklLayout(), numSplits);
  CHECK_EQ(status, E_SUCCESS);

  for (int i = 0; i < numConcats; i++) {
    this->input[i]->createMklLayout(
        this->forwardPrim, (dnnResourceType_t)(dnnResourceMultipleSrc + i));

    // TODO comes from caffe, it's different with others (DiffSrc/DiffDst)
    this->gradInput[i]->createMklLayout(
        this->backwardPrim, (dnnResourceType_t)(dnnResourceMultipleDst + i));
  }

  delete[] layouts;

  this->isFirstPass = false;
}

template <typename DType>
void MKLConcat<DType>::updateOutput(DType **input, DType *output)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  if (this->isFirstPass) firstPass();

  for (int i = 0; i < numConcats; i++) {
    this->input[i]->setUsrData(input[i]);
    this->input[i]->createConversion();
  }
  this->output->setUsrData(output);
  this->output->createConversion();

  dnnError_t status;
  void *resources[dnnResourceNumber];

  for (int i = 0; i < numConcats; i++) {
    resources[dnnResourceMultipleSrc + i] = this->input[i]->getConvertedData();
  }
  resources[dnnResourceDst] = this->output->getData();

  PERFSTART();
  status = dnnExecute<DType>(this->forwardPrim, resources);
  PERFEND("main computing");

  if (!this->output->isUseNext()) this->output->backToUsr();
}

template <typename DType>
void MKLConcat<DType>::updateGradInput(DType **gradInput, DType *gradOutput)
{
  caffe::cpu::OpenMpManager::setGpuDisabled();
  caffe::cpu::OpenMpManager::bindOpenMpThreads();

  for (int i = 0; i < numConcats; i++) {
    this->gradInput[i]->setUsrData(gradInput[i]);
    this->gradInput[i]->createConversion();
  }
  this->gradOutput->setUsrData(gradOutput);
  this->gradOutput->createConversion();

  dnnError_t status;
  void *resources[dnnResourceNumber];

  for (int i = 0; i < numConcats; i++) {
    resources[dnnResourceMultipleDst + i] = this->gradInput[i]->getData();
  }
  resources[dnnResourceSrc] = this->gradOutput->getConvertedData();

  PERFSTART();
  status = dnnExecute<DType>(this->backwardPrim, resources);
  PERFEND("main computing");

  for (int i = 0; i < numConcats; i++) {
    if (!this->gradInput[i]->isUsePrev()) this->gradInput[i]->backToUsr();
  }
}

template <typename ArrayType, typename DType>
jlong JNIConcatInit(JNIEnv *env, jclass thisClass, int numConcats,
                    int dimension, jintArray size)
{
  MKLConcat<DType> *ptr = new MKLConcat<DType>();

  jint *jSize =
      reinterpret_cast<int *>(env->GetPrimitiveArrayCritical(size, 0));
  ptr->init(numConcats, dimension, jSize);
  env->ReleasePrimitiveArrayCritical(size, jSize, 0);

  return reinterpret_cast<long>(ptr);
}

template <typename ArrayType, typename DType>
void JNIConcatUpdateOutput(JNIEnv *env, jclass thisClass, jobjectArray input,
                           jintArray inputOffset, ArrayType output,
                           jint outputOffset, long classPtr)
{
  MKLConcat<DType> *ptr = reinterpret_cast<MKLConcat<DType> *>(classPtr);

  jint *jInputOffset =
      reinterpret_cast<jint *>(env->GetPrimitiveArrayCritical(inputOffset, 0));

  // TODO we should re-write, this version makes a little complict.
  int len = env->GetArrayLength(input);
  DType *inputArrStart[len];
  DType *inputArr[len];
  ArrayType jInputArr[len];
  for (int i = 0; i < len; i++) {
    jInputArr[i]     = (ArrayType)(env->GetObjectArrayElement(input, i));
    inputArrStart[i] = reinterpret_cast<DType *>(
        env->GetPrimitiveArrayCritical(jInputArr[i], 0));
    inputArr[i] = inputArrStart[i] + jInputOffset[i];
  }

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutput(
      new ZipArray<ArrayType, DType>(env, output, outputOffset, ptr->output));

  ptr->updateOutput(inputArr, jOutput->getPtr());

  for (int i = 0; i < len; i++) {
    env->ReleasePrimitiveArrayCritical(jInputArr[i], inputArrStart[i], 0);
  }

  env->ReleasePrimitiveArrayCritical(inputOffset, jInputOffset, 0);
}

template <typename ArrayType, typename DType>
void JNIConcatUpdateGradInput(JNIEnv *env, jclass thisClass,
                              jobjectArray inputDiff, jintArray inputDiffOffset,
                              ArrayType outputDiff, jint outputDiffOffset,
                              long classPtr)
{
  MKLConcat<DType> *ptr = reinterpret_cast<MKLConcat<DType> *>(classPtr);

  jint *jInputDiffOffset = reinterpret_cast<jint *>(
      env->GetPrimitiveArrayCritical(inputDiffOffset, 0));

  int len = env->GetArrayLength(inputDiff);
  DType *inputDiffArrStart[len];
  DType *inputDiffArr[len];
  ArrayType jInputDiffArr[len];
  for (int i = 0; i < len; i++) {
    jInputDiffArr[i] = (ArrayType)(env->GetObjectArrayElement(inputDiff, i));
    inputDiffArrStart[i] = reinterpret_cast<DType *>(
        env->GetPrimitiveArrayCritical(jInputDiffArr[i], 0));
    inputDiffArr[i] = inputDiffArrStart[i] + jInputDiffOffset[i];
  }

  std::shared_ptr<ZipArray<ArrayType, DType>> jOutputDiff(
      new ZipArray<ArrayType, DType>(env, outputDiff, outputDiffOffset,
                                     ptr->gradOutput));

  ptr->updateGradInput(inputDiffArr, jOutputDiff->getPtr());

  for (int i = 0; i < len; i++) {
    env->ReleasePrimitiveArrayCritical(jInputDiffArr[i], inputDiffArrStart[i],
                                       0);
  }

  env->ReleasePrimitiveArrayCritical(inputDiffOffset, jInputDiffOffset, 0);
}

// Macro
#define ConcatInit(DType, JType, JArrayType)                                \
  JNIEXPORT                                                                 \
  jlong JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ConcatInit##DType( \
      JNIEnv *env, jclass thisClass, jint numConcats, jint dimension,       \
      jintArray size)                                                       \
  {                                                                         \
    return JNIConcatInit<JArrayType, JType>(env, thisClass, numConcats,     \
                                            dimension, size);               \
  }

#define ConcatForward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                   \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ConcatForward##DType( \
      JNIEnv *env, jclass thisClass, jobjectArray input,                      \
      jintArray inputOffset, JArrayType output, jint outputOffset,            \
      long classPtr)                                                          \
  {                                                                           \
    JNIConcatUpdateOutput<JArrayType, JType>(                                 \
        env, thisClass, input, inputOffset, output, outputOffset, classPtr);  \
  }

#define ConcatBackward(DType, JType, JArrayType)                               \
  JNIEXPORT                                                                    \
  void JNICALL Java_com_intel_analytics_sparkdl_mkl_MKL_ConcatBackward##DType( \
      JNIEnv *env, jclass thisClass, jobjectArray inputDiff,                   \
      jintArray inputDiffOffset, JArrayType outputDiff, jint outputDiffOffset, \
      long classPtr)                                                           \
  {                                                                            \
    JNIConcatUpdateGradInput<JArrayType, JType>(env, thisClass, inputDiff,     \
                                                inputDiffOffset, outputDiff,   \
                                                outputDiffOffset, classPtr);   \
  }

#ifdef __cplusplus
extern "C" {
#endif

// Double
ConcatInit(Double, jdouble, jdoubleArray);
ConcatForward(Double, jdouble, jdoubleArray);
ConcatBackward(Double, jdouble, jdoubleArray);

// Float
ConcatInit(Float, jfloat, jfloatArray);
ConcatForward(Float, jfloat, jfloatArray);
ConcatBackward(Float, jfloat, jfloatArray);

#ifdef __cplusplus
}
#endif
