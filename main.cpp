#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "NvInfer.h"

// result value check of cuda runtime
#define CHECK(call) check(call, __LINE__, __FILE__)

inline bool check(cudaError_t e, int iLine, const char *szFile) {
  if (e != cudaSuccess) {
    std::cout << "CUDA runtime API error " << cudaGetErrorName(e) << " at line " << iLine
              << " in file " << szFile << std::endl;
    return false;
  }
  return true;
}

using namespace nvinfer1;

// plugin debug function
#ifdef DEBUG
#define WHERE_AM_I()                      \
  do {                                    \
    printf("%14p[%s]\n", this, __func__); \
  } while (0);
#else
#define WHERE_AM_I()
#endif  // ifdef DEBUG

#define CEIL_DIVIDE(X, Y) (((X) + (Y) - 1) / (Y))
#define ALIGN_TO(X, Y) (CEIL_DIVIDE(X, Y) * (Y))

// TensorRT journal
class Logger : public ILogger {
 public:
  Severity reportableSeverity;

  Logger(Severity severity = Severity::kINFO) : reportableSeverity(severity) {}

  void log(Severity severity, const char *msg) noexcept override {
    if (severity > reportableSeverity) {
      return;
    }
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "VERBOSE: ";
        break;
    }
    std::cerr << msg << std::endl;
  }
};

template <typename T>
void printArrayRecursion(const T *pArray, Dims32 dim, int iDim, int iStart) {
  if (iDim == dim.nbDims - 1) {
    for (int i = 0; i < dim.d[iDim]; ++i) {
      std::cout << std::fixed << std::setprecision(3) << std::setw(6) << double(pArray[iStart + i])
                << " ";
    }
  } else {
    int nElement = 1;
    for (int i = iDim + 1; i < dim.nbDims; ++i) {
      nElement *= dim.d[i];
    }
    for (int i = 0; i < dim.d[iDim]; ++i) {
      printArrayRecursion<T>(pArray, dim, iDim + 1, iStart + i * nElement);
    }
  }
  std::cout << std::endl;
  return;
}

template <typename T>
void printArrayInfomation(const T *pArray, Dims32 dim, std::string name = std::string(""),
                          bool bPrintArray = false, int n = 10) {
  // print shape information
  std::cout << std::endl;
  std::cout << name << ": (";
  for (int i = 0; i < dim.nbDims; ++i) {
    std::cout << dim.d[i] << ", ";
  }
  std::cout << ")" << std::endl;

  // print statistic information
  int nElement = 1;  // number of elements with batch dimension
  for (int i = 0; i < dim.nbDims; ++i) {
    nElement *= dim.d[i];
  }

  double sum = double(pArray[0]);
  double absSum = double(fabs(double(pArray[0])));
  double sum2 = double(pArray[0]) * double(pArray[0]);
  double diff = 0.0;
  double maxValue = double(pArray[0]);
  double minValue = double(pArray[0]);
  for (int i = 1; i < nElement; ++i) {
    sum += double(pArray[i]);
    absSum += double(fabs(double(pArray[i])));
    sum2 += double(pArray[i]) * double(pArray[i]);
    maxValue = double(pArray[i]) > maxValue ? double(pArray[i]) : maxValue;
    minValue = double(pArray[i]) < minValue ? double(pArray[i]) : minValue;
    diff += abs(double(pArray[i]) - double(pArray[i - 1]));
  }
  double mean = sum / nElement;
  double var = sum2 / nElement - mean * mean;

  std::cout << "absSum=" << std::fixed << std::setprecision(4) << std::setw(7) << absSum << ",";
  std::cout << "mean=" << std::fixed << std::setprecision(4) << std::setw(7) << mean << ",";
  std::cout << "var=" << std::fixed << std::setprecision(4) << std::setw(7) << var << ",";
  std::cout << "max=" << std::fixed << std::setprecision(4) << std::setw(7) << maxValue << ",";
  std::cout << "min=" << std::fixed << std::setprecision(4) << std::setw(7) << minValue << ",";
  std::cout << "diff=" << std::fixed << std::setprecision(4) << std::setw(7) << diff << ",";
  std::cout << std::endl;

  // print first n element and last n element
  for (int i = 0; i < n; ++i) {
    std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
  }
  std::cout << std::endl;
  for (int i = nElement - n; i < nElement; ++i) {
    std::cout << std::fixed << std::setprecision(5) << std::setw(8) << double(pArray[i]) << ", ";
  }
  std::cout << std::endl;

  // print the whole array
  if (bPrintArray) {
    printArrayRecursion<T>(pArray, dim, 0, 0);
  }

  return;
}
template void printArrayInfomation(const float *, Dims32, std::string, bool, int);
template void printArrayInfomation(const int *, Dims32, std::string, bool, int);
template void printArrayInfomation(const bool *, Dims32, std::string, bool, int);

__inline__ size_t dataTypeToSize(DataType dataType) {
  switch ((int)dataType) {
    case int(DataType::kFLOAT):
      return 4;
    case int(DataType::kHALF):
      return 2;
    case int(DataType::kINT8):
      return 1;
    case int(DataType::kINT32):
      return 4;
    case int(DataType::kBOOL):
      return 1;
    default:
      return 4;
  }
}

__inline__ std::string dataTypeToString(DataType dataType) {
  switch (dataType) {
    case DataType::kFLOAT:
      return std::string("FP32 ");
    case DataType::kHALF:
      return std::string("FP16 ");
    case DataType::kINT8:
      return std::string("INT8 ");
    case DataType::kINT32:
      return std::string("INT32");
    case DataType::kBOOL:
      return std::string("BOOL ");
    default:
      return std::string("Unknown");
  }
}

__inline__ std::string shapeToString(Dims32 dim) {
  std::string output("(");
  if (dim.nbDims == 0) {
    return output + std::string(")");
  }
  for (int i = 0; i < dim.nbDims - 1; ++i) {
    output += std::to_string(dim.d[i]) + std::string(", ");
  }
  output += std::to_string(dim.d[dim.nbDims - 1]) + std::string(")");
  return output;
}

using namespace nvinfer1;

const std::string trtFile{"../data/identity_neural_network.trt"};
static Logger gLogger(ILogger::Severity::kERROR);

int main() {
  ICudaEngine *engine = nullptr;
  std::ifstream engineFile(trtFile, std::ios::binary);
  long int fsize = 0;

  engineFile.seekg(0, engineFile.end);
  fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);
  std::vector<char> engineString(fsize);
  engineFile.read(engineString.data(), fsize);
  if (engineString.size() == 0) {
    std::cout << "Failed getting serialized engine!" << std::endl;
    return -1;
  }
  std::cout << "Succeeded getting serialized engine!" << std::endl;

  IRuntime *runtime{createInferRuntime(gLogger)};
  engine = runtime->deserializeCudaEngine(engineString.data(), fsize);
  if (engine == nullptr) {
    std::cout << "Failed loading engine!" << std::endl;
    return -1;
  }
  std::cout << "Succeeded loading engine!" << std::endl;

  int nIO = engine->getNbIOTensors();
  int nInput = 0;
  int nOutput = 0;
  std::vector<std::string> vTensorName(nIO);
  for (int i = 0; i < nIO; ++i) {
    vTensorName[i] = std::string(engine->getIOTensorName(i));
    nInput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kINPUT);
    nOutput += int(engine->getTensorIOMode(vTensorName[i].c_str()) == TensorIOMode::kOUTPUT);
  }

  IExecutionContext *context = engine->createExecutionContext();

  for (int i = 0; i < nIO; ++i) {
    std::cout << std::string(i < nInput ? "Input [" : "Output[");
    std::cout << i << std::string("]-> ");
    std::cout << dataTypeToString(engine->getTensorDataType(vTensorName[i].c_str()))
              << std::string(" ");
    std::cout << shapeToString(engine->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
    std::cout << shapeToString(context->getTensorShape(vTensorName[i].c_str())) << std::string(" ");
    std::cout << vTensorName[i] << std::endl;
  }

  std::vector<int> vTensorSize(nIO, 0);
  for (int i = 0; i < nIO; ++i) {
    Dims32 dim = context->getTensorShape(vTensorName[i].c_str());
    int size = 1;
    for (int j = 0; j < dim.nbDims; ++j) {
      size *= dim.d[j];
    }
    vTensorSize[i] = size * dataTypeToSize(engine->getTensorDataType(vTensorName[i].c_str()));
  }

  std::vector<void *> vBufferH{nIO, nullptr};
  std::vector<void *> vBufferD{nIO, nullptr};
  for (int i = 0; i < nIO; ++i) {
    vBufferH[i] = (void *)new char[vTensorSize[i]];
    CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
  }

  float *pData = (float *)vBufferH[0];

  for (int i = 0;
       i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str()));
       ++i) {
    pData[i] = float(0);
  }
  for (int i = 0; i < nInput; ++i) {
    CHECK(cudaMemcpy(vBufferD[i], vBufferH[i], vTensorSize[i], cudaMemcpyHostToDevice));
  }

  for (int i = 0; i < nIO; ++i) {
    context->setTensorAddress(vTensorName[i].c_str(), vBufferD[i]);
  }

  context->enqueueV3(0);

  for (int i = nInput; i < nIO; ++i) {
    CHECK(cudaMemcpy(vBufferH[i], vBufferD[i], vTensorSize[i], cudaMemcpyDeviceToHost));
  }

  float *output_data = (float *)vBufferH[1];
  bool flag = true;
  for (int i = 0;
       i < vTensorSize[0] / dataTypeToSize(engine->getTensorDataType(vTensorName[0].c_str()));
       ++i) {
    if (output_data[i] - pData[i] > 1e-5) {
      flag = false;
      continue;
    }
  }
  if (flag) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  for (int i = 0; i < nIO; ++i) {
    delete[] vBufferH[i];
    CHECK(cudaFree(vBufferD[i]));
  }
  return 0;
}
