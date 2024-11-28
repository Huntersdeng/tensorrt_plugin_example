#include "identity_conv_plugin.h"

#include <cassert>
#include <exception>

namespace nvinfer1::plugin {

template <typename Type, typename BufferType>
void write(BufferType*& buffer, const Type& val) {
  static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
  *reinterpret_cast<Type*>(buffer) = val;
  buffer += sizeof(Type);
}

template <typename Type, typename BufferType>
Type read(const BufferType*& buffer) {
  static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
  Type val = *reinterpret_cast<const Type*>(buffer);
  buffer += sizeof(Type);
  return val;
}

IdentityConvPlugin::IdentityConvPlugin(IdentityConvParams params) : params_(params) {}

IdentityConvPlugin::IdentityConvPlugin(void const* data, size_t length) {
  deserialize(static_cast<const uint8_t*>(data), length);
}

void IdentityConvPlugin::deserialize(uint8_t const* data, size_t length) {
  uint8_t const* d = data;
  params_.group = read<int32_t>(d);
  params_.dtype = read<nvinfer1::DataType>(d);
  params_.channel_size = read<int32_t>(d);
  params_.height = read<int32_t>(d);
  params_.width = read<int32_t>(d);
  params_.dtype_bytes = read<size_t>(d);
  assert(d == data + length);
}

void IdentityConvPlugin::configurePlugin(PluginTensorDesc const* in, int32_t nbInput,
                                         PluginTensorDesc const* out, int32_t nbOutput) noexcept {
  assert(nbInput == 2);
  assert(nbOutput = 1);
  assert(in[0].dims.nbDims == 3);
  assert(out[0].dims.nbDims == 3);
  assert(in[0].dims.d[0] == out[0].dims.d[0]);
  assert(in[0].dims.d[1] == out[0].dims.d[1]);
  assert(in[0].dims.d[2] == out[0].dims.d[2]);
  assert(in[0].type == out[0].type);

  params_.channel_size = in[0].dims.d[0];
  params_.height = in[0].dims.d[1];
  params_.width = in[0].dims.d[2];
  params_.dtype = in[0].type;
  if (params_.dtype == DataType::kINT8) {
    params_.dtype_bytes = 1;
  } else if (params_.dtype == DataType::kHALF) {
    params_.dtype_bytes = 2;
  } else if (params_.dtype == DataType::kFLOAT) {
    params_.dtype_bytes = 4;
  } else {
    assert(false);
  }
}

int32_t IdentityConvPlugin::initialize() noexcept { return 0; }

void IdentityConvPlugin::terminate() noexcept {}

int32_t IdentityConvPlugin::getNbOutputs() const noexcept { return 1; }

Dims IdentityConvPlugin::getOutputDimensions(int32_t index, Dims const* inputs,
                                             int32_t nbInputDims) noexcept {
  assert(index == 0);
  assert(nbInputDims == 2);
  assert(inputs != nullptr);
  assert(inputs[0].nbDims == 3);

  Dims dims_output;
  dims_output.nbDims = inputs[0].nbDims;
  dims_output.d[0] = inputs[0].d[0];
  dims_output.d[1] = inputs[0].d[1];
  dims_output.d[2] = inputs[0].d[2];

  return dims_output;
}

nvinfer1::DataType IdentityConvPlugin::getOutputDataType(int32_t index,
                                                         nvinfer1::DataType const* inputTypes,
                                                         int32_t nbInputs) const noexcept {
  assert(index == 0);
  assert(nbInputs == 2);
  return inputTypes[0];
}

size_t IdentityConvPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept { return 0; }

size_t IdentityConvPlugin::getSerializationSize() const noexcept {
  return sizeof(int32_t) * 4 + sizeof(nvinfer1::DataType) + sizeof(size_t);
}

void IdentityConvPlugin::serialize(void* buffer) const noexcept {
  void* d = buffer;
  write(d, params_.group);
  write(d, params_.dtype);
  write(d, params_.channel_size);
  write(d, params_.height);
  write(d, params_.width);
  write(d, params_.dtype_bytes);
  assert(d == buffer + getSerializationSize());
}

void IdentityConvPlugin::destroy() noexcept { delete this; }

bool IdentityConvPlugin::supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                                   int32_t nbInputs,
                                                   int32_t nbOutputs) const noexcept {
  assert(nbInputs == 2);
  assert(nbOutputs == 1);
  assert(pos < nbInputs + nbOutputs);

  bool is_valid_combination = false;

  is_valid_combination |=
      (inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR);
  is_valid_combination |=
      (inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR);
  is_valid_combination &= (pos < nbInputs || (inOut[pos].format == inOut[0].format &&
                                              inOut[pos].type == inOut[0].type));
  return is_valid_combination;
}

AsciiChar const* IdentityConvPlugin::getPluginType() const noexcept {
  return kIDENTITY_CONV_PLUGIN_NAME;
}

AsciiChar const* IdentityConvPlugin::getPluginVersion() const noexcept {
  return kIDENTITY_CONV_PLUGIN_VERSION;
}

void IdentityConvPlugin::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
  plugin_namespace_ = pluginNamespace;
}

AsciiChar const* IdentityConvPlugin::getPluginNamespace() const noexcept {
  return plugin_namespace_;
}

bool IdentityConvPlugin::isOutputBroadcastAcrossBatch(int32_t outputIndex,
                                                      bool const* inputIsBroadcasted,
                                                      int32_t nbInputs) const noexcept {
  return false;
}

bool IdentityConvPlugin::canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept {
  return false;
}

int32_t IdentityConvPlugin::enqueue(int32_t batchSize, void const* const* inputs,
                                    void* const* outputs, void* workspace,
                                    cudaStream_t stream) noexcept {
  size_t const size_bytes =
      batchSize * params_.channel_size * params_.height * params_.width * params_.dtype_bytes;
  const cudaError_t status =
      cudaMemcpyAsync(outputs[0], inputs[0], size_bytes, cudaMemcpyDeviceToDevice, stream);
  return status;
}

IPluginV2Ext* IdentityConvPlugin::clone() const noexcept {
  try {
    IPluginV2IOExt* const plugin{new IdentityConvPlugin{params_}};
    plugin->setPluginNamespace(plugin_namespace_);
    return plugin;
  } catch (std::exception const& e) {
    assert(false);
  }
  return nullptr;
}

}  // namespace nvinfer1::plugin