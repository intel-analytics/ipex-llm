#include "utils.h"
#include <cstdio>
#include <cmath>
#include <iostream>

#if 0
int computeOut(int input, int pad, int kernel, int stride)
{
  // if (((input + 2 * pad - kernel) % stride) != 0)
  //   printf("%d %d %d %d\n", input, pad, kernel, stride);
  // TODO Should we substitute with ceil or floor when compute the output?
  //std::cout << static_cast<int>(ceil(static_cast<float>((input + 2 * pad - kernel) / stride) + 1)) << std::endl;
  //std::cout << ((input + 2 * pad - kernel) / stride) + 1 << std::endl;
  //return static_cast<int>(floor(static_cast<float>((input + 2 * pad - kernel) / stride) + 1));
  // return static_cast<int>(
  //    static_cast<float>((input + 2 * pad - kernel) / stride) + 1);
  //return ((input + 2 * pad - kernel) / stride) + 1;
  int tmp = ((input + 2 * pad - kernel) / stride) + 1;
  //if (((input + 2 * pad - kernel) % stride) != 0)
  //  tmp += 1;
  return tmp;
}
#endif

int computeOut(int input, int pad, int kernel, int stride, bool ceilMode)
{
  if (ceilMode) {
    return static_cast<int>(ceil(static_cast<float>(
          input + 2 * pad - kernel) / stride)) + 1;
  } else {
    return static_cast<int>(floor(static_cast<float>(
          input + 2 * pad - kernel) / stride)) + 1;
  }
}

#if 0
int main()
{
  std::cout << computeOut(4, 0, 3, 2, true);
  std::cout << computeOut(4, 0, 3, 2, false);

  std::cout << computeOut(3, 1, 2, 1, true);
  std::cout << computeOut(3, 1, 2, 1, false);

  return 0;
}
#endif
