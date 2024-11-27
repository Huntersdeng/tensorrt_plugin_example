#include "identity_conv_plugin_creator.h"

#include <cassert>
#include <cstring>
#include <sstream>

#include "identity_conv_plugin.h"
#include "utils.h"

namespace nvinfer1::plugin {

IdentityConvPluginCreator::IdentityConvPluginCreator() {
  plugin_attributes_.clear();
  plugin_attributes_.emplace_back("kernel_shape", nullptr, PluginFieldType::kINT32, 2);
  plugin_attributes_.emplace_back("strides", nullptr, PluginFieldType::kINT32, 2);
  plugin_attributes_.emplace_back("pads", nullptr, PluginFieldType::kINT32, 4);
  plugin_attributes_.emplace_back("group", nullptr, PluginFieldType::kINT32, 1);

  plugin_field_collection_.nbFields = plugin_attributes_.size();
  plugin_field_collection_.fields = plugin_attributes_.data();
}
AsciiChar const* IdentityConvPluginCreator::getPluginName() const noexcept {
  return kIDENTITY_CONV_PLUGIN_NAME;
}

AsciiChar const* IdentityConvPluginCreator::getPluginVersion() const noexcept {
  return kIDENTITY_CONV_PLUGIN_VERSION;
}

void IdentityConvPluginCreator::setPluginNamespace(AsciiChar const* pluginNamespace) noexcept {
  plugin_namespace_ = pluginNamespace;
}

AsciiChar const* IdentityConvPluginCreator::getPluginNamespace() const noexcept {
  return plugin_namespace_.c_str();
}

PluginFieldCollection const* IdentityConvPluginCreator::getFieldNames() noexcept {
  return &plugin_field_collection_;
}

IPluginV2* IdentityConvPluginCreator::createPlugin(AsciiChar const* name,
                                                   PluginFieldCollection const* fc) noexcept {
  try {
    nvinfer1::PluginField const* fields = fc->fields;
    int32_t nbFields = fc->nbFields;

    assert(nbFields == 4);

    std::vector<int32_t> kernel_shape;
    std::vector<int32_t> strides;
    std::vector<int32_t> pads;
    int32_t group;

    for (int32_t i = 0; i < nbFields; ++i) {
      if (strcmp(fields[i].name, "kernel_shape") == 0) {
        assert(fields[i].type == PluginFieldType::kINT32);
        kernel_shape.assign(static_cast<int32_t const*>(fields[i].data),
                            static_cast<int32_t const*>(fields[i].data) + fields[i].length);
      } else if (strcmp(fields[i].name, "strides") == 0) {
        assert(fields[i].type == PluginFieldType::kINT32);
        strides.assign(static_cast<int32_t const*>(fields[i].data),
                       static_cast<int32_t const*>(fields[i].data) + fields[i].length);
      } else if (strcmp(fields[i].name, "pads") == 0) {
        assert(fields[i].type == PluginFieldType::kINT32);
        pads.assign(static_cast<int32_t const*>(fields[i].data),
                    static_cast<int32_t const*>(fields[i].data) + fields[i].length);
      } else if (strcmp(fields[i].name, "group") == 0) {
        assert(fields[i].type == PluginFieldType::kINT32);
        group = *static_cast<int32_t const*>(fields[i].data);
      }
    }

    std::stringstream ss;
    ss << "Plugin Attributes:";
    logInfo(ss.str().c_str());

    ss.str("");
    ss << "kernel_shape: ";
    for (auto const& val : kernel_shape) {
      ss << val << " ";
    }
    logInfo(ss.str().c_str());

    ss.str("");
    ss << "strides: ";
    for (auto const& val : strides) {
      ss << val << " ";
    }
    logInfo(ss.str().c_str());

    ss.str("");
    ss << "pads: ";
    for (auto const& val : pads) {
      ss << val << " ";
    }
    logInfo(ss.str().c_str());

    ss.str("");
    ss << "group: " << group;
    logInfo(ss.str().c_str());

    IdentityConvParams params;
    params.group = group;

    IdentityConvPlugin* plugin = new IdentityConvPlugin(params);
    plugin->setPluginNamespace(plugin_namespace_.c_str());
    return plugin;

  } catch (std::exception const& e) {
    return nullptr;
  }
  return nullptr;
}

IPluginV2* IdentityConvPluginCreator::deserializePlugin(AsciiChar const* name,
                                                        void const* serialData,
                                                        size_t serialLength) noexcept {
  try {
    IdentityConvPlugin* plugin = new IdentityConvPlugin(serialData, serialLength);
    plugin->setPluginNamespace(plugin_namespace_.c_str());
    return plugin;
  } catch (std::exception const& e) {
    return nullptr;
  }
  return nullptr;
}

// REGISTER_TENSORRT_PLUGIN(IdentityConvPluginCreator);

}  // namespace nvinfer1::plugin