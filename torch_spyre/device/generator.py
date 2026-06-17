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

"""Utility functions for handling Spyre generators."""

import torch
import torch_spyre._C as _C


def is_spyre_generator(generator) -> bool:
    """Check if generator is a Spyre device generator.

    Args:
        generator: A torch.Generator or None

    Returns:
        True if generator is not None and is on Spyre device, False otherwise
    """
    return generator is not None and generator.device.type == "spyre"


def convert_generator_to_cpu(generator):
    """Convert a Spyre generator to CPU generator with synced state.

    Creates a new CPU generator and copies the state from the Spyre generator
    to maintain consistency in random number generation.

    Args:
        generator: A Spyre generator to convert

    Returns:
        A CPU generator with the same state as the input Spyre generator

    Raises:
        RuntimeError: If state conversion fails
    """
    cpu_generator = torch.Generator(device="cpu")
    try:
        spyre_state = generator.get_state()
        cpu_state = _C.spyre_state_to_cpu_state(spyre_state)
        cpu_generator.set_state(cpu_state)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert Spyre generator state to CPU: {e}"
        ) from e
    return cpu_generator


def sync_generator_state_to_spyre(cpu_generator, spyre_generator):
    """Sync updated CPU generator state back to Spyre generator.

    After using a CPU generator for operations, this function syncs the
    updated state back to the original Spyre generator to maintain
    consistency across devices.

    Args:
        cpu_generator: CPU generator with updated state
        spyre_generator: Spyre generator to update

    Raises:
        RuntimeError: If state conversion or sync fails
    """
    try:
        updated_cpu_state = cpu_generator.get_state()
        updated_spyre_state = _C.cpu_state_to_spyre_state(updated_cpu_state)
        spyre_generator.set_state(updated_spyre_state)
    except Exception as e:
        raise RuntimeError(
            f"Failed to sync CPU generator state back to Spyre: {e}"
        ) from e
