#include <NvInferRuntime.h>

#include <cstring>
#include <sstream>

void caughtError(std::exception const& e) {
  getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void logInfo(char const* msg) { getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg); }

void reportAssertion(bool success, char const* msg, char const* file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    std::abort();
  }
}

void reportValidation(bool success, char const* msg, char const* file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }
}