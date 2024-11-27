#include <ai_engine.h>

#include <iostream>
#include <map>

#include "identity_conv_plugin_creator.h"

using ai_star::alg::common::AIBuff;
using ai_star::alg::common::AIEngine;
using ai_star::alg::common::AIEngineConfig;

int main() {
  AIEngineConfig config;
  config.model_path = "../data/identity_neural_network.trt";
  config.input_len["X0"] = 3 * 480 * 960;
  config.output_len["X3"] = 3 * 480 * 960;
  config.input_shape["X0"] = std::vector<int64_t>{1, 3, 480, 960};
  config.output_shape["X3"] = std::vector<int64_t>{1, 3, 480, 960};

  AIEngine engine;
  engine.Init(config);

  std::map<std::string, AIBuff> input_map, output_map;

  float* input_data = (float*)malloc(config.input_len["X0"] * sizeof(float));
  for (int i = 0; i < config.input_len["X0"]; i++) {
    input_data[i] = i;
  }

  input_map["X0"] =
      AIBuff(reinterpret_cast<uint8_t*>(input_data), config.input_len["X0"] * sizeof(float));
  output_map["X3"] = AIBuff();
  output_map["X3"].resize(config.output_len["X3"] * sizeof(float));

  engine.DoInference(input_map, output_map);

  float* output_data = (float*)output_map["X3"].data();
  bool flag = true;
  for (int i = 0; i < config.output_len["X3"]; i++) {
    if (output_data[i] - input_data[i] > 1e-5) {
      flag = false;
      continue;
    }
  }
  if (flag) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  return 0;
}