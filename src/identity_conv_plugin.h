#pragma once

#include <NvInferPlugin.h>

constexpr char const* const kIDENTITY_CONV_PLUGIN_NAME{"IdentityConv"};
constexpr char const* const kIDENTITY_CONV_PLUGIN_VERSION{"1"};

namespace nvinfer1::plugin {

struct IdentityConvParams {
  int32_t group;
  nvinfer1::DataType dtype;
  int32_t channel_size;
  int32_t height;
  int32_t width;
  size_t dtype_bytes;
};

class IdentityConvPlugin : public IPluginV2IOExt {
 public:
  IdentityConvPlugin() = default;
  ~IdentityConvPlugin() override = default;
  IdentityConvPlugin(IdentityConvParams params);
  IdentityConvPlugin(const void* data, size_t length);
  virtual void configurePlugin(PluginTensorDesc const* in, int32_t nbInput,
                               PluginTensorDesc const* out, int32_t nbOutput) noexcept override;
  virtual bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                         int32_t nbInputs,
                                         int32_t nbOutputs) const noexcept override;

  virtual nvinfer1::DataType getOutputDataType(int32_t index, nvinfer1::DataType const* inputTypes,
                                               int32_t nbInputs) const noexcept override;

  IPluginV2Ext* clone() const noexcept override;

  virtual AsciiChar const* getPluginType() const noexcept override;

  virtual AsciiChar const* getPluginVersion() const noexcept override;

  virtual int32_t getNbOutputs() const noexcept override;

  virtual Dims getOutputDimensions(int32_t index, Dims const* inputs,
                                   int32_t nbInputDims) noexcept override;

  virtual int32_t initialize() noexcept override;

  virtual void terminate() noexcept override;

  virtual size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

  virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs,
                          void* workspace, cudaStream_t stream) noexcept override;

  virtual size_t getSerializationSize() const noexcept override;

  virtual void serialize(void* buffer) const noexcept override;

  virtual void destroy() noexcept override;

  virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;

  virtual AsciiChar const* getPluginNamespace() const noexcept override;

  virtual bool isOutputBroadcastAcrossBatch(int32_t outputIndex, bool const* inputIsBroadcasted,
                                            int32_t nbInputs) const noexcept override;

  virtual bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

 private:
  void deserialize(uint8_t const* data, size_t length);

  IdentityConvParams params_;
  char const* plugin_namespace_;
};
}  // namespace nvinfer1::plugin