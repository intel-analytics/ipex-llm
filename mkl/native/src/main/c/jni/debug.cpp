#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include "debug.h"

LogMessage::LogMessage(const char *file, int line, LogType type)
{
  int len = strlen(file) + 20;
  char *buf = new char[len];
  type_ = type;

  const char *lastSlash = strrchr(file, '/');
  const char *fileName = (lastSlash == NULL) ? file : lastSlash + 1;

  snprintf(buf, len, "%c %s %s:%d] ", "DIWEFI"[type], "MKL", fileName, line);
  stream() << buf;

  delete[] buf;
}

LogMessage::~LogMessage()
{
  stream() << std::endl;
  if (type_ == FATAL) {
    stream() << "Aborting..." << std::endl;
    abort();
  }
}

std::ostream& LogMessage::stream()
{
  if (type_ >= WARNNING) {
    return std::cerr;
  } else {
    return std::cout;
  }
}
