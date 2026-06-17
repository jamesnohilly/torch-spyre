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

"""Tests for SpyreGenerator implementation."""

import pytest
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

import torch_spyre._C as _C
from torch_spyre.device.generator import (
    is_spyre_generator,
)


@instantiate_parametrized_tests
class TestSpyreGenerator(TestCase):
    """Test suite for SpyreGenerator functionality."""

    def setUp(self):
        super().setUp()
        # Reset to a known seed for reproducibility
        torch.manual_seed(42)

    def test_generator_creation_default(self):
        """Test creating a default Spyre generator."""
        gen = torch.Generator(device="spyre")
        assert gen.device.type == "spyre"
        assert gen is not None

    def test_generator_creation_with_seed(self):
        """Test creating a Spyre generator with a specific seed."""
        seed = 12345
        gen = torch.Generator(device="spyre")
        gen.manual_seed(seed)

        # Verify the seed was set
        initial_seed = gen.initial_seed()
        assert initial_seed == seed

    def test_generator_device_type(self):
        """Test that generator reports correct device type."""
        gen = torch.Generator(device="spyre")
        assert gen.device.type == "spyre"
        assert is_spyre_generator(gen)

    def test_generator_device_index(self):
        """Test generator with specific device index."""
        if torch.spyre.device_count() > 0:
            gen = torch.Generator(device="spyre:0")
            assert gen.device.type == "spyre"
            assert gen.device.index == 0

    def test_manual_seed(self):
        """Test manual_seed sets the seed correctly."""
        gen = torch.Generator(device="spyre")
        seed = 42
        gen.manual_seed(seed)

        # Generate some random numbers
        x1 = torch.randn(10, device="spyre", generator=gen)

        # Reset to same seed
        gen.manual_seed(seed)
        x2 = torch.randn(10, device="spyre", generator=gen)

        # Should produce identical results
        torch.testing.assert_close(x1, x2)

    def test_current_seed(self):
        """Test current_seed returns the set seed."""
        gen = torch.Generator(device="spyre")
        seed = 7777
        gen.manual_seed(seed)

        current = gen.initial_seed()
        assert current == seed

    def test_get_state(self):
        """Test get_state returns a valid state tensor."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(123)

        state = gen.get_state()

        # State should be a CPU byte tensor
        assert state.device.type == "cpu"
        assert state.dtype == torch.uint8
        assert state.numel() > 0

    def test_set_state(self):
        """Test set_state restores generator state."""
        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(456)

        # Generate some numbers and save state
        _ = torch.randn(5, device="spyre", generator=gen1)
        state = gen1.get_state()
        x2 = torch.randn(5, device="spyre", generator=gen1)

        # Create new generator and restore state
        gen2 = torch.Generator(device="spyre")
        gen2.set_state(state)
        x3 = torch.randn(5, device="spyre", generator=gen2)

        # x3 should match x2 (same state after x1)
        torch.testing.assert_close(x2, x3)

    def test_state_persistence(self):
        """Test that state can be saved and restored across generators."""
        seed = 888
        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(seed)

        # Generate sequence
        _ = [torch.randn(3, device="spyre", generator=gen1) for _ in range(3)]
        state = gen1.get_state()

        # Continue generating
        seq2 = [torch.randn(3, device="spyre", generator=gen1) for _ in range(3)]

        # Restore state and regenerate
        gen1.set_state(state)
        seq3 = [torch.randn(3, device="spyre", generator=gen1) for _ in range(3)]

        # seq3 should match seq2
        for t2, t3 in zip(seq2, seq3):
            torch.testing.assert_close(t2, t3)

    def test_clone_generator(self):
        """Test generator cloning via state save/restore."""
        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(111)

        # "Clone" by saving and restoring state
        state = gen1.get_state()
        gen2 = torch.Generator(device="spyre")
        gen2.set_state(state)

        # Both should produce same sequence initially
        x1 = torch.randn(5, device="spyre", generator=gen1)
        x2 = torch.randn(5, device="spyre", generator=gen2)
        torch.testing.assert_close(x1, x2)

        # After advancing gen1, they should diverge
        _ = torch.randn(5, device="spyre", generator=gen1)
        y1 = torch.randn(5, device="spyre", generator=gen1)
        y2 = torch.randn(5, device="spyre", generator=gen2)

        # y1 and y2 should be different
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y1, y2)

    def test_cpu_state_to_spyre_state(self):
        """Test conversion from CPU generator state to Spyre state."""
        # Create CPU generator with known state
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(555)
        cpu_state = cpu_gen.get_state()

        # Convert to Spyre state
        spyre_state = _C.cpu_state_to_spyre_state(cpu_state)

        # Spyre state should be smaller (no normal distribution cache)
        assert spyre_state.numel() < cpu_state.numel()
        assert spyre_state.dtype == torch.uint8

    def test_spyre_state_to_cpu_state(self):
        """Test conversion from Spyre generator state to CPU state."""
        # Create Spyre generator with known state
        spyre_gen = torch.Generator(device="spyre")
        spyre_gen.manual_seed(666)
        spyre_state = spyre_gen.get_state()

        # Convert to CPU state
        cpu_state = _C.spyre_state_to_cpu_state(spyre_state)

        # CPU state should be larger (includes normal distribution cache)
        assert cpu_state.numel() > spyre_state.numel()
        assert cpu_state.dtype == torch.uint8

    def test_randn_with_generator(self):
        """Test torch.randn with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(100)

        x = torch.randn(10, 10, device="spyre", generator=gen, dtype=torch.float16)

        assert x.shape == (10, 10)
        assert x.device.type == "spyre"
        assert x.dtype == torch.float16

        # Check that values are reasonable (not all zeros or identical)
        x_cpu = x.cpu()
        assert not (x_cpu == 0).all().item()
        assert not (x_cpu == x_cpu[0, 0]).all().item()

    def test_randn_reproducibility(self):
        """Test that randn with same seed produces same results."""
        seed = 200

        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(seed)
        x1 = torch.randn(20, device="spyre", generator=gen1, dtype=torch.float16)

        gen2 = torch.Generator(device="spyre")
        gen2.manual_seed(seed)
        x2 = torch.randn(20, device="spyre", generator=gen2, dtype=torch.float16)

        torch.testing.assert_close(x1, x2)

    def test_uniform_with_generator(self):
        """Test tensor.uniform_() with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(300)

        x = torch.empty(100, device="spyre", dtype=torch.float16)
        x.uniform_(generator=gen)

        x_cpu = x.cpu()
        # Values should be in [0, 1)
        assert (x_cpu >= 0.0).all().item()
        assert (x_cpu < 1.0).all().item()
        # Should not all be identical
        assert not (x_cpu == x_cpu[0]).all().item()

    def test_uniform_custom_range_with_generator(self):
        """Test tensor.uniform_(from, to) with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(400)

        x = torch.empty(100, device="spyre", dtype=torch.float16)
        x.uniform_(-10.0, 10.0, generator=gen)

        x_cpu = x.cpu()
        # Values should be in [-10, 10)
        assert (x_cpu >= -10.0).all().item()
        assert (x_cpu < 10.0).all().item()

    def test_random_custom_range_with_generator(self):
        """Test tensor.random_(from, to) with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(600)

        x = torch.empty(50, device="spyre", dtype=torch.float16)
        x.random_(-5, 5, generator=gen)

        x_cpu = x.cpu()
        # Values should be in [-5, 5)
        assert (x_cpu >= -5).all().item()
        assert (x_cpu < 5).all().item()


if __name__ == "__main__":
    run_tests()
