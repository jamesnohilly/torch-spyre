# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for host compute metadata validation."""
import json

import pytest
import torch
import torch_spyre

@pytest.fixture(scope="module", autouse=True)
def initialize_runtime():
    """Initialize Spyre runtime before running tests."""
    # Initialize torch with spyre device to start runtime
    torch.zeros(1, device="spyre")
    yield


class TestHostComputeMetadataValidation:
    """Test suite for validateHostComputeMetadata function."""

    def test_valid_metadata_passes(self):
        """Test that valid metadata with all required fields passes validation."""
        valid_metadata = {
            "vdci": {"field1": "value1", "field2": 123},
            "senConstants": [
                {"name": "const1", "value": 1.0},
                {"name": "const2", "value": 2.0},
            ],
        }
        metadata_str = json.dumps(valid_metadata)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert result.ok(), f"Expected validation to pass, but got errors: {[e.message for e in result.errors]}"
        assert len(result.errors) == 0, "Expected no validation errors"

    def test_empty_metadata_fails(self):
        """Test that empty metadata fails validation with clear error message."""
        empty_metadata = "{}"

        result = torch_spyre._C.validate_host_compute_metadata(empty_metadata)

        assert not result.ok(), "Expected validation to fail for empty metadata"
        assert len(result.errors) == 1
        assert "empty" in result.errors[0].message.lower()

    def test_missing_vdci_field_fails(self):
        """Test that missing 'vdci' field fails validation with clear error."""
        metadata_missing_vdci = {"senConstants": [{"name": "const1", "value": 1.0}]}
        metadata_str = json.dumps(metadata_missing_vdci)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail for missing 'vdci'"
        assert len(result.errors) == 1
        assert "vdci" in result.errors[0].message
        assert "missing" in result.errors[0].message.lower()

    def test_missing_senconstants_field_fails(self):
        """Test that missing 'senConstants' field fails validation with clear error."""
        metadata_missing_senconstants = {"vdci": {"field1": "value1"}}
        metadata_str = json.dumps(metadata_missing_senconstants)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail for missing 'senConstants'"
        assert len(result.errors) == 1
        assert "senConstants" in result.errors[0].message.lower()
        assert "missing" in result.errors[0].message.lower()

    def test_missing_both_fields_fails(self):
        """Test that missing both required fields produces two error messages."""
        metadata_missing_both = {"other_field": "value"}
        metadata_str = json.dumps(metadata_missing_both)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail for missing both fields"
        assert len(result.errors) == 2
        error_messages = [e.message for e in result.errors]
        assert any("vdci" in msg for msg in error_messages)
        assert any("senconstants" in msg for msg in error_messages)

    def test_malformed_json_fails(self):
        """Test that malformed JSON fails validation with clear error."""
        malformed_json = '{"vdci": "unclosed'

        with pytest.raises(Exception) as exc_info:
            torch_spyre._C.validate_host_compute_metadata(malformed_json)

        # The JSON parsing error should be raised before validation
        assert "parse" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()

    def test_non_object_json_fails(self):
        """Test that non-object JSON (e.g., array, string) fails validation."""
        json_array = json.dumps(["item1", "item2"])

        result = torch_spyre._C.validate_host_compute_metadata(json_array)

        assert not result.ok(), "Expected validation to fail for non-object JSON"
        assert len(result.errors) == 1
        assert "object" in result.errors[0].message.lower()

    def test_vdci_not_object_fails(self):
        """Test that 'vdci' field not being an object fails validation."""
        metadata_vdci_not_object = {
            "vdci": "not_an_object",
            "senConstants": [{"name": "const1", "value": 1.0}],
        }
        metadata_str = json.dumps(metadata_vdci_not_object)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail when 'vdci' is not an object"
        assert len(result.errors) == 1
        assert "vdci" in result.errors[0].message
        assert "object" in result.errors[0].message.lower()

    def test_senconstants_not_array_fails(self):
        """Test that 'senConstants' field not being an array fails validation."""
        metadata_senconstants_not_array = {
            "vdci": {"field1": "value1"},
            "senConstants": {"const1": 1.0},
        }
        metadata_str = json.dumps(metadata_senconstants_not_array)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail when 'senConstants' is not an array"
        assert len(result.errors) == 1
        assert "senConstants" in result.errors[0].message.lower()
        assert "array" in result.errors[0].message.lower()

    def test_both_fields_wrong_types_fails(self):
        """Test that both fields having wrong types produces two error messages."""
        metadata_both_wrong_types = {"vdci": "string", "senConstants": 123}
        metadata_str = json.dumps(metadata_both_wrong_types)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail when both fields have wrong types"
        assert len(result.errors) == 2
        error_messages = [e.message for e in result.errors]
        assert any("vdci" in msg and "object" in msg.lower() for msg in error_messages)
        assert any("senConstants" in msg.lower() and "array" in msg.lower() for msg in error_messages)

    def test_unexpected_fields_fail(self):
        """Test that unexpected fields beyond required ones are rejected."""
        metadata_with_extra = {
            "vdci": {"field1": "value1"},
            "senConstants": [{"name": "const1", "value": 1.0}],
            "extra_field": "extra_value",
            "another_extra": {"nested": "data"},
        }
        metadata_str = json.dumps(metadata_with_extra)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert not result.ok(), "Expected validation to fail with unexpected fields"
        assert len(result.errors) == 2
        error_messages = [e.message for e in result.errors]
        assert any("extra_field" in msg and "unexpected" in msg.lower() for msg in error_messages)
        assert any("another_extra" in msg and "unexpected" in msg.lower() for msg in error_messages)

    def test_empty_required_fields_pass(self):
        """Test that empty object/array for required fields still pass validation."""
        metadata_empty_fields = {"vdci": {}, "senConstants": []}
        metadata_str = json.dumps(metadata_empty_fields)

        result = torch_spyre._C.validate_host_compute_metadata(metadata_str)

        assert result.ok(), f"Expected validation to pass with empty fields, but got errors: {[e.message for e in result.errors]}"
        assert len(result.errors) == 0
