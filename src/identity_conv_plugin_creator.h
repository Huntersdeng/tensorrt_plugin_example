#pragma once

#include <NvInferPlugin.h>

#include <string>
#include <vector>

namespace nvinfer1::plugin {
class IdentityConvPluginCreator : public IPluginCreator {
 public:
  IdentityConvPluginCreator();
  virtual ~IdentityConvPluginCreator() = default;

  virtual AsciiChar const* getPluginName() const noexcept override;

  virtual AsciiChar const* getPluginVersion() const noexcept override;

  virtual PluginFieldCollection const* getFieldNames() noexcept override;

  virtual IPluginV2* createPlugin(AsciiChar const* name,
                                  PluginFieldCollection const* fc) noexcept override;

  virtual IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData,
                                       size_t serialLength) noexcept override;

  virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept override;

  virtual AsciiChar const* getPluginNamespace() const noexcept override;

 private:
  std::string plugin_namespace_;
  std::vector<PluginField> plugin_attributes_;
  PluginFieldCollection plugin_field_collection_;
};
}  // namespace nvinfer1::plugin