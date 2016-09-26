#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <iostream>

const int DBG = 0, INFO = 1, WARNNING = 2, ERROR = 3, FATAL = 4, DEFALT = 5;
typedef int LogType;

class LogMessage
{
 public:
  LogMessage(const char *file, int line, LogType type);
  ~LogMessage();
  std::ostream &stream();

 private:
  LogType type_;
};

#define CHECK(x) \
  if (!(x))      \
    LogMessage(__FILE__, __LINE__, WARNNING).stream() << "Check failed " #x;

//#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_EQ(x, y)                              \
  if (!((x) == (y)))                                \
  LogMessage(__FILE__, __LINE__, WARNNING).stream() \
      << "Check failed. " #x << " = " << x << ",which should be " #y
#define CHECK_NE(x, y) CHECK((x) != (y))

#define LOG(x) LogMessage(__FILE__, __LINE__, x).stream()

#ifdef PERF
const int INPERF = 1;
#else
const int INPERF = 0;
#endif

#define PERFSTART()                           \
  do {                                        \
    struct timespec start, end;               \
    if (INPERF) {                             \
      clock_gettime(CLOCK_MONOTONIC, &start); \
    }

#define PERFEND(msg)                                                  \
  if (INPERF) {                                                       \
    clock_gettime(CLOCK_MONOTONIC, &end);                             \
    LOG(INFO) << __func__ << " " << msg << " costs: "                 \
              << (end.tv_sec - start.tv_sec) * 1000 +                 \
                     (double)(end.tv_nsec - start.tv_nsec) / 1000000; \
  }                                                                   \
  }                                                                   \
  while (0)                                                           \
    ;

/**
 * @brief print 4 dimensions data
 *
 * Because the input/output is orgnized as vector, it should be more human
 * readable when we debug the result generated.
 *
 * @param input input/output data which is orgnized as vecotr/array.
 * @param num how many images
 * @param channel how many channels, like 3
 * @param height image height
 * @param width image width
 * @param msg messge user defined
 */
template <typename Type>
void printData(Type *input, size_t num, size_t channel, size_t height,
               size_t width, const char *msg)
{
  std::cout << std::string(msg) << " CHECK IN CPP..." << std::endl;

  for (int i = 0; i < num; i++) {
    std::cout << "The " << i << " num." << std::endl;
    for (int j = 0; j < channel; j++) {
      std::cout << "The " << j << " channel." << std::endl;
      for (int k = 0; k < height; k++) {
        for (int t = 0; t < width; t++) {
          int index = ((i * channel + j) * height + k) * width + t;
          std::cout << input[index] << '\t';
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

#endif
