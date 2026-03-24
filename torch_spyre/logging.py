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

import logging
import os
import torch._logging._internal as pytorch_logging

logger = logging.getLogger(__name__)


def _setup_logger():

    pytorch_logging.register_log("spyre_kernel_launch", [
                                 "torch_spyre._inductor.spyre_kernel"])
    pytorch_logging.register_log("spyre_compiler", [
        "torch_spyre.execution.async_compile",
    ])


def _get_env_bool(var_name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(var_name, str(int(default)))
    return value.lower() in ("1", "true", "yes", "on")
