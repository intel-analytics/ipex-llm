#ifndef _MKL_LAYER_H
#define _MKL_LAYER_H
#include <memory>

#include "MKLWrapper.h"
#include "memory.h"

template <typename DType>
class MKLLayer
{
 public:
  MKLLayer();
  ~MKLLayer();

  static void setPrev(long prev, long curr);

  void init(size_t inputNumber, size_t inputChannel, size_t inputHeight,
            size_t inputWidth, size_t dimension);

  std::shared_ptr<MKLData<DType>> input, output, gradInput, gradOutput;

  int dimension;

  // parameters of pooling layer
  size_t inputSize[4];
  size_t inputStrides[4];

  // If it's the first pass, we should create some conversions.
  // After that, we need not do that again.
  // Default is true.
  //
  // Note:
  //   1. Defaultly, we assume that the address of input will not change.
  //   2. The address of input is real address of Array in JVM.
  //   3. TODO It will set to false after an iteration (forward and backward).
  bool isFirstPass;

  dnnPrimitive_t forwardPrim, backwardPrim;
};

template <typename DType>
void MKLLayer<DType>::init(size_t inputNumber, size_t inputChannel,
                           size_t inputHeight, size_t inputWidth,
                           size_t dimension)
{
  inputSize[0] = inputWidth;
  inputSize[1] = inputHeight;
  inputSize[2] = inputChannel;
  inputSize[3] = inputNumber;

  this->dimension = dimension;

  inputStrides[0] = 1;
  for (int i = 1; i < 4; i++) {
    inputStrides[i] = inputStrides[i - 1] * inputSize[i - 1];
  }

  input->createUsrLayout(dimension, inputSize, inputStrides);
  gradInput->createUsrLayout(dimension, inputSize, inputStrides);
}

template <typename DType>
MKLLayer<DType>::MKLLayer()
    : input(new MKLData<DType>()),
      output(new MKLData<DType>()),
      gradInput(new MKLData<DType>()),
      gradOutput(new MKLData<DType>()),
      isFirstPass(true),
      forwardPrim(NULL),
      backwardPrim(NULL)
{
}

template <typename DType>
MKLLayer<DType>::~MKLLayer()
{
  if (forwardPrim) {
    dnnDelete<DType>(forwardPrim);
    forwardPrim = NULL;
  }

  if (backwardPrim) {
    dnnDelete<DType>(backwardPrim);
    backwardPrim = NULL;
  }
}

template <typename DType>
void MKLLayer<DType>::setPrev(long prev, long curr)
{
  MKLLayer<DType> *prevLayer = reinterpret_cast<MKLLayer<DType> *>(prev);
  MKLLayer<DType> *currLayer = reinterpret_cast<MKLLayer<DType> *>(curr);

  dnnLayout_t prevLayout = prevLayer->gradOutput->getMklLayout();
  dnnLayout_t currLayout = currLayer->gradInput->getMklLayout();

  if (dnnLayoutCompare<DType>(prevLayout, currLayout)) {
    prevLayer->gradOutput->setUseNext(true);
    prevLayer->gradOutput = currLayer->gradInput;
    currLayer->gradInput->setUsePrev(true);
  }

  prevLayout = prevLayer->output->getMklLayout();
  currLayout = currLayer->input->getMklLayout();

  if (dnnLayoutCompare<DType>(prevLayout, currLayout)) {
    prevLayer->output->setUseNext(true);
    currLayer->input = prevLayer->output;
    currLayer->input->setUsePrev(true);
  }
}
#endif
