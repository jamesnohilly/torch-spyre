/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hcm_validator.h"

#include <string>

namespace spyre {

namespace {

// Helper function to validate that a required field exists and is a JSON object
void validateRequiredObjectField(ValidationResult& result,
                                 const nlohmann::json& json,
                                 const char* field_name) {
  if (!json.contains(field_name)) {
    result.add("Missing required field '" + std::string(field_name) +
               "' in metadata");
  } else if (!json[field_name].is_object()) {
    result.add("Expected a JSON object for '" + std::string(field_name) + "'");
  }
}

// Helper function to validate that a required field exists and is a JSON array
void validateRequiredArrayField(ValidationResult& result,
                                const nlohmann::json& json,
                                const char* field_name) {
  if (!json.contains(field_name)) {
    result.add("Missing required field '" + std::string(field_name) +
               "' in metadata");
  } else if (!json[field_name].is_array()) {
    result.add("Expected a JSON array for '" + std::string(field_name) + "'");
  }
}

}  // anonymous namespace

ValidationResult validateHostComputeMetadata(const nlohmann::json& metadata) {
  ValidationResult result;

  // Validate that the given metadata is not empty.
  if (metadata.empty()) {
    result.add("Given metadata is empty.");
    return result;
  }

  // Validate that the given metadata is parseable JSON.
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(metadata);
  }
  catch (const nlohmann::json::parse_error& e) {
    result.add("Malformed JSON: " + std::string(e.what()));
    return result;
  }

  // Validate that the parsed JSON is an object.
  if (!parsed.is_object()) {
    result.add("Expected a JSON object");
    return result;
  }

  // Check for unexpected fields and validate known fields inline.
  for (const auto& [key, value] : parsed.items()) {
    if (key != "vdci" && key != "senConstants") {
      result.add("Unexpected field '" + key + "' in metadata");
    }
  }

  // Validate required fields are present and have correct types.
  validateRequiredObjectField(result, parsed, "vdci");
  validateRequiredArrayField(result, parsed, "senConstants");

  return result;
}

}  // namespace spyre
