class VectorUnit:
    def __init__(
        self,
        total_vector_flops_per_cycle,
        word_size,
        flops_per_exp,
        vector_width,
        vector_count,
    ):
        self.total_vector_flops_per_cycle = total_vector_flops_per_cycle
        self.word_size = word_size
        self.flops_per_exp = flops_per_exp
        self.vector_width = vector_width
        self.vector_count = vector_count
        denom = vector_width * vector_count
        self.flops_per_cycle = (total_vector_flops_per_cycle + denom - 1) // denom


class SystolicArray:
    def __init__(
        self,
        array_height,
        array_width,
        mac_per_cycle,
        input_word_size,
        output_word_size,
    ):
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size


class Core:
    def __init__(self, vector_unit, systolic_array, systolic_array_count, SRAM_size):
        self.vector_unit = vector_unit
        self.systolic_array = systolic_array
        self.systolic_array_count = systolic_array_count
        self.SRAM_size = SRAM_size
        self.vector_word_size = vector_unit.word_size


class Overhead:
    def __init__(self, matmul, softmax, layernorm, gelu):
        self.matmul = matmul
        self.softmax = softmax
        self.layernorm = layernorm
        self.gelu = gelu


class ComputeModule:
    def __init__(
        self,
        core,
        core_count,
        clock_freq,
        l2_size,
        l2_bandwidth_per_cycle,
        overhead,
    ):
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle
        self.total_vector_flops_per_cycle = (
            core.vector_unit.total_vector_flops_per_cycle * core_count
        )
        self.total_vector_flops = self.total_vector_flops_per_cycle * clock_freq
        self.total_systolic_array_flops = (
            core_count
            * core.systolic_array_count
            * core.systolic_array.mac_per_cycle
            * 2
            * core.systolic_array.array_height
            * core.systolic_array.array_width
            * clock_freq
        )
        self.overhead = overhead


class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency


class MemoryModule:
    def __init__(self, memory_capacity):
        self.memory_capacity = memory_capacity


class Device:
    def __init__(self, compute_module, io_module, memory_module):
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module


# A100 scalar parameters
A100_CORE_COUNT = 108
A100_CLOCK_FREQ_HZ = 1.41e9
A100_L2_SIZE_BYTES = 40 * 1024**2
A100_L2_BW_BYTES_PER_CYCLE = 5120
A100_IO_BW_BYTES_PER_SEC = 2039e9
A100_IO_LATENCY_SEC = 1e-6
A100_MEMORY_BYTES = 80e9

# A100 vector/tensor core parameters
A100_VECTOR_UNIT_FP16 = VectorUnit(512, 2, 35, 32, 4)
A100_SYSTOLIC_ARRAY_FP16 = SystolicArray(16, 16, 1, 2, 2)
A100_CORE_FP16 = Core(
    vector_unit=A100_VECTOR_UNIT_FP16,
    systolic_array=A100_SYSTOLIC_ARRAY_FP16,
    systolic_array_count=4,
    SRAM_size=192 * 1024,
)
A100_OVERHEAD = Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5)

# A100 module objects
A100_COMPUTE_MODULE_FP16 = ComputeModule(
    core=A100_CORE_FP16,
    core_count=A100_CORE_COUNT,
    clock_freq=A100_CLOCK_FREQ_HZ,
    l2_size=A100_L2_SIZE_BYTES,
    l2_bandwidth_per_cycle=A100_L2_BW_BYTES_PER_CYCLE,
    overhead=A100_OVERHEAD,
)
A100_IO_MODULE = IOModule(
    bandwidth=A100_IO_BW_BYTES_PER_SEC,
    latency=A100_IO_LATENCY_SEC,
)
A100_MEMORY_MODULE_80GB = MemoryModule(memory_capacity=A100_MEMORY_BYTES)
A100_80GB_FP16 = Device(
    compute_module=A100_COMPUTE_MODULE_FP16,
    io_module=A100_IO_MODULE,
    memory_module=A100_MEMORY_MODULE_80GB,
)

# Dict-style access for compatibility.
device_dict = {"A100_80GB_fp16": A100_80GB_FP16}

__all__ = [
    "VectorUnit",
    "SystolicArray",
    "Core",
    "Overhead",
    "ComputeModule",
    "IOModule",
    "MemoryModule",
    "Device",
    "A100_CORE_COUNT",
    "A100_CLOCK_FREQ_HZ",
    "A100_L2_SIZE_BYTES",
    "A100_L2_BW_BYTES_PER_CYCLE",
    "A100_IO_BW_BYTES_PER_SEC",
    "A100_IO_LATENCY_SEC",
    "A100_MEMORY_BYTES",
    "A100_VECTOR_UNIT_FP16",
    "A100_SYSTOLIC_ARRAY_FP16",
    "A100_CORE_FP16",
    "A100_OVERHEAD",
    "A100_COMPUTE_MODULE_FP16",
    "A100_IO_MODULE",
    "A100_MEMORY_MODULE_80GB",
    "A100_80GB_FP16",
    "device_dict",
]
