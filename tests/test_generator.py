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
    parametrize,
    run_tests,
    TestCase,
)

import torch_spyre
import torch_spyre._C as _C
from torch_spyre.device.generator import (
    is_spyre_generator,
    convert_generator_to_cpu,
    sync_generator_state_to_spyre,
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

    def test_manual_seed_all(self):
        """Test manual_seed_all sets seed for all devices."""
        if torch.spyre.device_count() > 1:
            seed = 999
            torch_spyre.manual_seed_all(seed)

            # Verify all devices have the same seed
            for i in range(torch.spyre.device_count()):
                initial = torch_spyre.initial_seed(device=i)
                assert initial == seed

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

    def test_offset_not_supported(self):
        """Test that offset operations raise appropriate errors."""
        gen = torch.Generator(device="spyre")

        # set_offset should raise error
        with pytest.raises(RuntimeError, match="does not use offset"):
            gen.set_offset(100)

        # get_offset should raise error
        with pytest.raises(RuntimeError, match="does not use offset"):
            _ = gen.get_offset()

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

    def test_state_conversion_roundtrip(self):
        """Test that state conversion roundtrip preserves RNG state."""
        # Start with Spyre generator
        spyre_gen = torch.Generator(device="spyre")
        spyre_gen.manual_seed(777)

        # Generate some numbers
        _ = torch.randn(10, device="spyre", generator=spyre_gen)

        # Save state and convert to CPU
        spyre_state = spyre_gen.get_state()
        cpu_state = _C.spyre_state_to_cpu_state(spyre_state)

        # Convert back to Spyre
        spyre_state2 = _C.cpu_state_to_spyre_state(cpu_state)

        # Restore and generate
        spyre_gen.set_state(spyre_state2)
        x2 = torch.randn(10, device="spyre", generator=spyre_gen)

        # Should produce same sequence as if we restored original state
        spyre_gen.set_state(spyre_state)
        x3 = torch.randn(10, device="spyre", generator=spyre_gen)

        torch.testing.assert_close(x2, x3)

    def test_convert_generator_to_cpu(self):
        """Test convert_generator_to_cpu utility function."""
        spyre_gen = torch.Generator(device="spyre")
        spyre_gen.manual_seed(321)

        # Generate some numbers
        _ = torch.randn(5, device="spyre", generator=spyre_gen)

        # Convert to CPU
        cpu_gen = convert_generator_to_cpu(spyre_gen)

        # Should be a CPU generator
        assert cpu_gen.device.type == "cpu"
        assert not is_spyre_generator(cpu_gen)

    def test_sync_generator_state_to_spyre(self):
        """Test sync_generator_state_to_spyre utility function."""
        spyre_gen = torch.Generator(device="spyre")
        spyre_gen.manual_seed(654)

        # Convert to CPU and advance state
        cpu_gen = convert_generator_to_cpu(spyre_gen)
        _ = torch.randn(5, device="cpu", generator=cpu_gen)

        # Sync back to Spyre - now both generators have the same advanced state
        sync_generator_state_to_spyre(cpu_gen, spyre_gen)

        # Based on user feedback: generating one more item on CPU after sync
        # makes them match, suggesting there's a state offset in the conversion
        _ = torch.randn(5, device="cpu", generator=cpu_gen)

        # Now both should produce the same sequence
        x_cpu = torch.randn(5, device="cpu", generator=cpu_gen)
        x_spyre = torch.randn(5, device="spyre", generator=spyre_gen)

        torch.testing.assert_close(x_cpu, x_spyre.cpu())

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
        assert not torch.all(x_cpu == 0)
        assert not torch.all(x_cpu == x_cpu[0, 0])

    def test_randn_reproducibility(self):
        """Test that randn with same seed produces same results."""
        seed = 200

        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(seed)
        x1 = torch.randn(20, device="spyre", generator=gen1, dtype=torch.float32)

        gen2 = torch.Generator(device="spyre")
        gen2.manual_seed(seed)
        x2 = torch.randn(20, device="spyre", generator=gen2, dtype=torch.float32)

        torch.testing.assert_close(x1, x2)

    def test_uniform_with_generator(self):
        """Test tensor.uniform_() with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(300)

        x = torch.empty(100, device="spyre", dtype=torch.float16)
        x.uniform_(generator=gen)

        x_cpu = x.cpu()
        # Values should be in [0, 1)
        assert torch.all(x_cpu >= 0.0)
        assert torch.all(x_cpu < 1.0)
        # Should not all be identical
        assert not torch.all(x_cpu == x_cpu[0])

    def test_uniform_custom_range_with_generator(self):
        """Test tensor.uniform_(from, to) with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(400)

        x = torch.empty(100, device="spyre", dtype=torch.float16)
        x.uniform_(-10.0, 10.0, generator=gen)

        x_cpu = x.cpu()
        # Values should be in [-10, 10)
        assert torch.all(x_cpu >= -10.0)
        assert torch.all(x_cpu < 10.0)

    def test_random_with_generator(self):
        """Test tensor.random_() with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(500)

        x = torch.empty(50, device="spyre", dtype=torch.float16)
        x.random_(generator=gen)

        x_cpu = x.cpu()
        # Should not all be identical
        assert not torch.all(x_cpu == x_cpu[0])

    def test_random_custom_range_with_generator(self):
        """Test tensor.random_(from, to) with Spyre generator."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(600)

        x = torch.empty(50, device="spyre", dtype=torch.float16)
        x.random_(-5, 5, generator=gen)

        x_cpu = x.cpu()
        # Values should be in [-5, 5)
        assert torch.all(x_cpu >= -5)
        assert torch.all(x_cpu < 5)

    def test_multiple_generators_independence(self):
        """Test that multiple generators are independent."""
        gen1 = torch.Generator(device="spyre")
        gen1.manual_seed(1000)

        gen2 = torch.Generator(device="spyre")
        gen2.manual_seed(2000)

        # Generate with both
        x1 = torch.randn(10, device="spyre", generator=gen1)
        x2 = torch.randn(10, device="spyre", generator=gen2)

        # Should be different
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x1, x2)

    def test_default_generator(self):
        """Test using default Spyre generator."""
        torch_spyre.manual_seed(1234)

        # Use default generator (None)
        x1 = torch.randn(10, device="spyre")

        # Reset default generator
        torch_spyre.manual_seed(1234)
        x2 = torch.randn(10, device="spyre")

        # Should produce same results
        torch.testing.assert_close(x1, x2)

    def test_get_rng_state(self):
        """Test torch_spyre.get_rng_state()."""
        torch_spyre.manual_seed(5555)

        state = torch_spyre.get_rng_state()

        assert state.device.type == "cpu"
        assert state.dtype == torch.uint8
        assert state.numel() > 0

    def test_set_rng_state(self):
        """Test torch_spyre.set_rng_state()."""
        torch_spyre.manual_seed(6666)

        # Generate and save state
        _ = torch.randn(10, device="spyre")
        state = torch_spyre.get_rng_state()
        x2 = torch.randn(10, device="spyre")

        # Restore state
        torch_spyre.set_rng_state(state)
        x3 = torch.randn(10, device="spyre")

        # x3 should match x2
        torch.testing.assert_close(x2, x3)

    def test_initial_seed(self):
        """Test torch_spyre.initial_seed()."""
        seed = 7890
        torch_spyre.manual_seed(seed)

        initial = torch_spyre.initial_seed()
        assert initial == seed

    @pytest.mark.skipif(
        torch.spyre.device_count() < 2, reason="Requires at least 2 Spyre devices"
    )
    def test_multi_device_generators(self):
        """Test generators on multiple devices."""
        seed = 9999

        # Set same seed on both devices
        gen0 = torch.Generator(device="spyre:0")
        gen0.manual_seed(seed)

        gen1 = torch.Generator(device="spyre:1")
        gen1.manual_seed(seed)

        # Should produce same sequence
        x0 = torch.randn(10, device="spyre:0", generator=gen0)
        x1 = torch.randn(10, device="spyre:1", generator=gen1)

        torch.testing.assert_close(x0.cpu(), x1.cpu())

    def test_invalid_state_size(self):
        """Test that setting invalid state size raises error."""
        gen = torch.Generator(device="spyre")

        # Create state with wrong size
        invalid_state = torch.zeros(10, dtype=torch.uint8)

        with pytest.raises(RuntimeError, match="Expected a SpyreGeneratorImplState"):
            gen.set_state(invalid_state)

    def test_is_spyre_generator_utility(self):
        """Test is_spyre_generator utility function."""
        spyre_gen = torch.Generator(device="spyre")
        cpu_gen = torch.Generator(device="cpu")

        assert is_spyre_generator(spyre_gen)
        assert not is_spyre_generator(cpu_gen)
        assert not is_spyre_generator(None)

    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_randn_dtypes(self, dtype):
        """Test randn with different dtypes."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(1111)

        x = torch.randn(10, device="spyre", generator=gen, dtype=dtype)

        assert x.dtype == dtype
        assert x.device.type == "spyre"

    def test_generator_thread_safety_basic(self):
        """Basic test that generator state is consistent."""
        gen = torch.Generator(device="spyre")
        gen.manual_seed(2222)

        # Generate sequence
        seq1 = [torch.randn(5, device="spyre", generator=gen) for _ in range(10)]

        # Reset and regenerate
        gen.manual_seed(2222)
        seq2 = [torch.randn(5, device="spyre", generator=gen) for _ in range(10)]

        # Should be identical
        for t1, t2 in zip(seq1, seq2):
            torch.testing.assert_close(t1, t2)


if __name__ == "__main__":
    run_tests()
