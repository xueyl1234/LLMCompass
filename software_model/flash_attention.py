import os
import sys

# Allow running this file directly: `python software_model/matmul0.py`.
if __package__ is None or __package__ == "":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, data_type_dict
from math import ceil, log2, floor
from typing import Literal
import time
import numpy as np
import pandas as pd
from scalesim.scale_sim import scalesim
import copy

MatmulType = Literal["QK", "SV"]

class BatchedMatmul(Operator):
    def __init__(
        self,
        B: int,
        M: int,
        K: int,
        N: int,
        data_type: DataType,
        matmul_type: MatmulType,
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.B = B
        self.M = M
        self.K = K
        self.N = N
        if matmul_type not in ("QK", "SV"):
            raise ValueError(f"Unsupported matmul_type: {matmul_type}")
        self.matmul_type = matmul_type

    def compile_and_simulate(self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ) -> int:
        matmul = Matmul(self.M, self.K, self.N, self.data_type, self.matmul_type)
        matmul_cycle_count1 = (
            matmul.compile_and_simulate(pcb_module, compile_mode) * self.B
        ) # 方案A：每个batch单独计算，完全流水化，理想情况下latency是单个batch的计算时间乘以batch size。matmul_latency1 = bs * latency of Matmul(M,K,N)

        matmul = Matmul(
            self.M, self.K * self.B, self.N, self.data_type, self.matmul_type
        )
        if self.matmul_type == "QK":
            output_write_cycle_count = 0 # QK不需要写回最终结果
        elif self.matmul_type == "SV":
            output_write_cycle_count = ceil(
                (self.B - 1)
                * self.M
                * self.N
                * self.data_type.word_size
                / (
                    pcb_module.io_module.bandwidth
                    / pcb_module.compute_module.clock_freq
                )
            ) # SV需要写回
        else:
            raise ValueError(f"Unsupported matmul_type: {self.matmul_type}")
        matmul_cycle_count2 = (
            matmul.compile_and_simulate(pcb_module, compile_mode)
            + output_write_cycle_count
        ) # 方案B：把batch维度和K维度合并成一个大矩阵，计算一次得到所有batch的结果，计算时间是单次大矩阵乘法的时间加上把结果写回内存的时间（因为结果更大了，所以写回时间也增加了）。latency of Matmul(M, K*bs, N) + IO time of writing output (M*N*bs elements) back to memory
        self.best_cycle_count = min(matmul_cycle_count1, matmul_cycle_count2)
        self.best_latency = self.best_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        return self.best_cycle_count


class Matmul(Operator): # MNK指M*K的矩阵与K*N的矩阵相乘，输出M*N的矩阵。class Matmul(Operator):指复用Operator类的属性和方法，Matmul是Operator的子类，继承了Operator的属性和方法
    def __init__(
        self, M: int, K: int, N: int, data_type: DataType, matmul_type: MatmulType
    ):
        super().__init__(0, 0, 0, 0, data_type)
        self.M = M
        self.K = K
        self.N = N
        if matmul_type not in ("QK", "SV"):
            raise ValueError(f"Unsupported matmul_type: {matmul_type}")
        self.matmul_type = matmul_type
        self.output_shape = [self.M, self.N]
        self.look_up_table = None
        self.best_mapping = None
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.M * self.K * self.N
        self.io_count = self.M * self.K + self.K * self.N + self.M * self.N

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(loop_M: int, loop_N: int, loop_K: int, loop_order: str): # k必须在最内层循环，为了保证flash attention的正确运行
        assert loop_order in ["mnk", "nmk"]
        if loop_order == "mnk":
            for m in range(loop_M):
                for n in range(loop_N):
                    for k in range(loop_K):
                        yield m, n, k
        elif loop_order == "nmk":
            for n in range(loop_N):
                for m in range(loop_M):
                    for k in range(loop_K):
                        yield m, n, k

    class ComputationalGraph:
        def __init__(self, M: int, N: int, K: int, data_type: DataType):
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

        def display(self):
            print("-" * 10 + " Computational Graph " + "-" * 10)
            print(
                f"M: {self.M}, N: {self.N}, K: {self.K}, word_size(B): {self.data_type.word_size}"
            )

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_M_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            matmul_type: MatmulType = "QK",
            dataflow: str = "os",
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_M_tiling_factor = l0_M_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            if matmul_type not in ("QK", "SV"):
                raise ValueError(f"Unsupported matmul_type: {matmul_type}")
            self.matmul_type = matmul_type
            self.dataflow = dataflow

        def display(self):
            print(f'{"-"*10} Mapping {"-"*10}')
            print(
                f"l2_tile_M: {self.l2_tile_M}, l2_tile_N: {self.l2_tile_N}, l2_tile_K: {self.l2_tile_K}, is_l2_double_buffering: {self.is_l2_double_buffering}, l2_loop_order: {self.l2_loop_order}"
            )
            print(
                f"l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, l1_tile_K: {self.l1_tile_K}, l1_loop_order: {self.l1_loop_order}"
            )
            print(
                f"l0_M_tiling_factor: {self.l0_M_tiling_factor}, l0_N_tiling_factor: {self.l0_N_tiling_factor}, l0_K_tiling_factor: {self.l0_K_tiling_factor}"
            )

    @staticmethod
    def find_permutations(n): # 为 l0_M_tiling_factor, l0_N_tiling_factor, l0_K_tiling_factor 提供候选分解（把总并行资源 n 按 M/N/K 三个方向分配），n 是 systolic_array_count
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations) # 最终返回所有满足 i * j * k = n 的正整数三元组，用于把并行资源分配到 M/N/K 三个方向。

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "exhaustive",
    ) -> int:
        # 搜索最优mapping对应最小cycle
        min_cycle_count = 2**63 - 1
        best_mapping = None
        M = self.computational_graph.M
        N = self.computational_graph.N
        K = self.computational_graph.K
        if (M == 1 or N == 1) and (
            compile_mode == "heuristic-GPU"
            or compile_mode == "heuristic-our-throughput"
        ): # GEMV场景的heuristic快速计算，默认GEMV几乎都是io bound，就不做复杂的计算建模
            working_set_size = M * K + N * K + M * N
            total_io_count = working_set_size * self.data_type.word_size
            io_latency = total_io_count / pcb_module.io_module.bandwidth
            total_flop_count = 2 * M * N * K
            compute_latency = (
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                / pcb_module.compute_module.core_count
                / pcb_module.compute_module.clock_freq
            )
            self.best_mapping = None
            self.best_cycle_count = ceil(
                max(compute_latency, io_latency)
                * pcb_module.compute_module.clock_freq
            )
            self.best_latency = (
                self.best_cycle_count / pcb_module.compute_module.clock_freq
            )
            self.latency = self.best_latency
            return self.best_cycle_count
        if compile_mode == "exhaustive":
            # exhaustive: 全参数穷举
            for l2_tile_M_log2 in range(1, ceil(log2(self.computational_graph.M)) + 1):
                l2_tile_M = 2**l2_tile_M_log2 # l2_tile_M，l2中M的tile size，取2的整数倍次方为了缩减搜索空间
                for l2_tile_N_log2 in range(
                    5, ceil(log2(self.computational_graph.N)) + 1
                ):
                    l2_tile_N = 2**l2_tile_N_log2 # l2_tile_N，l2中N的tile size，取2的整数倍次方为了缩减搜索空间
                    for l2_tile_K_log2 in range(
                        5, ceil(log2(self.computational_graph.K)) + 1
                    ):
                        l2_tile_K = 2**l2_tile_K_log2 # l2_tile_K，l2中M的tile size，取2的整数倍次方为了缩减搜索空间
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue # l2可以单缓冲，所以将working set size限制在l2大小，而不是l2大小的1/2
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True # l2双缓冲判断
                        else:
                            is_l2_double_buffering = False
                        for l1_tile_M_log2 in range(1, l2_tile_M_log2 + 1):
                            l1_tile_M = 2**l1_tile_M_log2 # 缩小搜索空间
                            for l1_tile_N_log2 in range(5, l2_tile_N_log2 + 1):
                                l1_tile_N = 2**l1_tile_N_log2 # 缩小搜索空间
                                for l1_tile_K_log2 in range(5, l2_tile_K_log2 + 1):
                                    l1_tile_K = 2**l1_tile_K_log2 # 缩小搜索空间
                                    if (
                                        l1_tile_M * l1_tile_N
                                        + l1_tile_N * l1_tile_K
                                        + l1_tile_M * l1_tile_K
                                        > pcb_module.compute_module.core.SRAM_size
                                        // self.data_type.word_size
                                        // 2
                                    ):
                                        continue # l1必须双缓冲，所以将working set size of l1限制在l1大小的1/2
                                    for l2_loop_order in ["mnk", "nmk"]:
                                        for l1_loop_order in ["mnk", "nmk"]:
                                            for (
                                                l0_M_tiling_factor,
                                                l0_N_tiling_factor,
                                                l0_K_tiling_factor,
                                            ) in self.find_permutations(
                                                pcb_module.compute_module.core.systolic_array_count
                                            ): # l0 是脉动阵列层级的并行计算划分策略，必须是systolic array count的约数
                                                mapping = self.Mapping(
                                                    l2_tile_M,
                                                    l2_tile_N,
                                                    l2_tile_K,
                                                    is_l2_double_buffering,
                                                    l1_tile_M,
                                                    l1_tile_N,
                                                    l1_tile_K,
                                                    l2_loop_order,
                                                    l1_loop_order,
                                                    l0_M_tiling_factor,
                                                    l0_N_tiling_factor,
                                                    l0_K_tiling_factor,
                                                    self.matmul_type,
                                                )
                                                cycle_count = self.simulate(
                                                    self.computational_graph,
                                                    mapping,
                                                    pcb_module,
                                                )
                                                if cycle_count < min_cycle_count:
                                                    min_cycle_count = cycle_count
                                                    best_mapping = mapping
        elif compile_mode == "heuristic-our-throughput":
            # heuristic-our-throughput: 吞吐导向候选集
            i = 0
            for l2_tile_M in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
                for l2_tile_N in [
                    l2_tile_M // 4,
                    l2_tile_M // 2,
                    l2_tile_M,
                    l2_tile_M * 2,
                    l2_tile_M * 4,
                    l2_tile_M * 8,
                    l2_tile_M * 16,
                    l2_tile_M * 32,
                    
                ]:
                    l2_tile_K_max = (
                        pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                        - l2_tile_M * l2_tile_N
                    ) // (l2_tile_M + l2_tile_N)
                    if l2_tile_K_max < 1:
                        continue
                    l2_tile_K = min(l2_tile_K_max, K)
                    l2_tile_K = floor(log2(l2_tile_K))
                    l2_tile_K = 2**l2_tile_K
                    working_set_size = (
                        l2_tile_N * l2_tile_K
                        + l2_tile_M * l2_tile_K
                        + l2_tile_M * l2_tile_N
                    )
                    if (
                        working_set_size
                        > pcb_module.compute_module.l2_size // self.data_type.word_size
                    ):
                        continue
                    elif (
                        working_set_size
                        <= pcb_module.compute_module.l2_size
                        // self.data_type.word_size
                        // 2
                    ):
                        is_l2_double_buffering = True
                    else:
                        is_l2_double_buffering = False

                    assert is_l2_double_buffering

                    for l1_tile_M in [32, 64, 128, 256]:
                        l1_tile_M = min(l1_tile_M, l2_tile_M, l2_tile_N)
                        # if l1_tile_M > min(l2_tile_M, l2_tile_N):
                        #     continue
                        l1_tile_N = l1_tile_M
                        l1_tile_K_max = (
                            pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                            - l1_tile_M * l1_tile_N
                        ) // (l1_tile_M + l1_tile_N)
                        if l1_tile_K_max < 1:
                            continue
                        l1_tile_K = min(l1_tile_K_max, l2_tile_K)
                        l1_tile_K = floor(log2(l1_tile_K))
                        l1_tile_K = 2**l1_tile_K

                        if (
                            l1_tile_M * l1_tile_N
                            + l1_tile_N * l1_tile_K
                            + l1_tile_M * l1_tile_K
                            > pcb_module.compute_module.core.SRAM_size
                            // self.data_type.word_size
                            // 2
                        ):
                            continue
                        l2_loop_order = "mnk"
                        l1_loop_order = "mnk"
                        for (
                            l0_M_tiling_factor,
                            l0_N_tiling_factor,
                            l0_K_tiling_factor,
                        ) in [(2, 2, 1)]:
                            # self.find_permutations(
                            #     pcb_module.compute_module.core.systolic_array_count
                            # ):
                            i += 1
                            # start = time.time()
                            mapping = self.Mapping(
                                l2_tile_M,
                                l2_tile_N,
                                l2_tile_K,
                                is_l2_double_buffering,
                                l1_tile_M,
                                l1_tile_N,
                                l1_tile_K,
                                l2_loop_order,
                                l1_loop_order,
                                l0_M_tiling_factor,
                                l0_N_tiling_factor,
                                l0_K_tiling_factor,
                                self.matmul_type,
                            )
                            cycle_count = self.simulate(
                                self.computational_graph,
                                mapping,
                                pcb_module,
                            )
                            # end = time.time()
                            # if i % 1000 == 0:
                            #     print(f"{i} simulation time: {end-start}")
                            if cycle_count < min_cycle_count:
                                min_cycle_count = cycle_count
                                best_mapping = mapping
        elif compile_mode == "heuristic-GPU":
            # heuristic-GPU: A100经验候选集
            i = 0
            for l2_tile_M in [64, 128, 256, 512, 1024, 2048]:
                for l2_tile_N in [l2_tile_M // 2, l2_tile_M, l2_tile_M * 2]:
                    if K <= 12288:
                        l2_K_tiling_factor_list = [1, 2, 4, 8]
                    else:
                        l2_K_tiling_factor_list = [
                            K // 1024,
                            K // 2048,
                            K // 4096,
                            K // 8192,
                        ]
                    for l2_K_tiling_factor in l2_K_tiling_factor_list:
                        l2_tile_K = ceil(
                            self.computational_graph.K / l2_K_tiling_factor
                        )
                        l2_tile_K = 2 ** floor(log2(l2_tile_K))
                        working_set_size = (
                            l2_tile_N * l2_tile_K
                            + l2_tile_M * l2_tile_K
                            + l2_tile_M * l2_tile_N
                        )
                        if (
                            working_set_size
                            > pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                        ):
                            continue
                        elif (
                            working_set_size
                            <= pcb_module.compute_module.l2_size
                            // self.data_type.word_size
                            // 2
                        ):
                            is_l2_double_buffering = True
                        else:
                            is_l2_double_buffering = False

                        for l1_tile_M in [32, 64, 128, 256]:
                            if l1_tile_M > min(l2_tile_M, l2_tile_N):
                                continue
                            l1_tile_N = l1_tile_M
                            for l1_K_tiling_factor in [1, 2, 4, 8, 16, 32]:
                                l1_tile_K = ceil(l2_tile_K / l1_K_tiling_factor)
                                if (
                                    l1_tile_M * l1_tile_N
                                    + l1_tile_N * l1_tile_K
                                    + l1_tile_M * l1_tile_K
                                    > pcb_module.compute_module.core.SRAM_size
                                    // self.data_type.word_size
                                    // 2
                                ):
                                    continue
                                l2_loop_order = "mnk"
                                l1_loop_order = "mnk"
                                for (
                                    l0_M_tiling_factor,
                                    l0_N_tiling_factor,
                                    l0_K_tiling_factor,
                                ) in self.find_permutations(
                                    pcb_module.compute_module.core.systolic_array_count
                                ):
                                    i += 1
                                    start = time.time()
                                    mapping = self.Mapping(
                                        l2_tile_M,
                                        l2_tile_N,
                                        l2_tile_K,
                                        is_l2_double_buffering,
                                        l1_tile_M,
                                        l1_tile_N,
                                        l1_tile_K,
                                        l2_loop_order,
                                        l1_loop_order,
                                        l0_M_tiling_factor,
                                        l0_N_tiling_factor,
                                        l0_K_tiling_factor,
                                        self.matmul_type,
                                    )
                                    cycle_count = self.simulate(
                                        self.computational_graph,
                                        mapping,
                                        pcb_module,
                                    )
                                    end = time.time()
                                    # if i % 1000 == 0:
                                    #     print(f"{i} simulation time: {end-start}")
                                    if cycle_count < min_cycle_count:
                                        min_cycle_count = cycle_count
                                        best_mapping = mapping
            # print("total dse times:", i)
        else:
            raise ValueError(f"compile_mode {compile_mode} not supported")
        # 记录全局最优结果
        self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.best_cycle_count

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int: # 注解，表明返回值是int
        if self.look_up_table is None: # None表示表格未加载，需要初始化读取表格
            # 懒加载脉动阵列查找表
            column_names = [
                "M",
                "N",
                "K",
                "ArrayHeight",
                "ArrayWidth",
                "Dataflow",
                "cycle_count",
                "util_rate",
            ]
            lut_path = (
                f"./systolic_array_model/look_up_table_"
                f"{pcb_module.compute_module.core.systolic_array.array_height}_"
                f"{pcb_module.compute_module.core.systolic_array.array_width}.csv"
            )
            if not os.path.exists(lut_path):
                os.makedirs(os.path.dirname(lut_path), exist_ok=True)
                pd.DataFrame(columns=column_names).to_csv(
                    lut_path, header=False, index=False
                )
            try:
                self.look_up_table = pd.read_csv(
                    lut_path,
                    header=None,
                    names=column_names,
                )
            except pd.errors.EmptyDataError:
                self.look_up_table = pd.DataFrame(columns=column_names)
            self.look_up_table.drop_duplicates(
                inplace=True,
                subset=["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
            )
            # self.look_up_table.reset_index(drop=True, inplace=True)
            # self.look_up_table.to_csv(
            #     f"./systolic_array_model/look_up_table_{pcb_module.compute_module.core.systolic_array.array_height}_{pcb_module.compute_module.core.systolic_array.array_width}.csv",
            #     header=False,
            #     index=False,
            # )
            self.look_up_table.set_index(
                ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow"],
                inplace=True,
            )
        # print(self.look_up_table)
        # print(self.look_up_table.loc[(32, 16, 256, 16, 16, 'os'), "cycle_count"
        #                              ].item())
        # print('sdfsdfsdfsd')
        # exit()
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_M = mapping.l2_tile_M
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            )
        else:
            assert (
                l2_tile_M * l2_tile_N + l2_tile_N * l2_tile_K + l2_tile_M * l2_tile_K
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        M_l2_t = M // l2_tile_M
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        M_remain = M % l2_tile_M
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K
        # 按mapping切分L2 tile网格
        l2_tiles = np.empty(
            [ceil(M / l2_tile_M), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        ) # 这里创建一个空的numpy数组，元素类型是L2TileSimulator实例
        # print('-'*20)
        # print(l2_tiles.shape)
        if M_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:M_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            ) # 该切片（数组的子块）中的每个元素都变成一个 L2TileSimulator 实例（同一组参数）
        if M_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:M_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:M_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_M,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                M_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                M_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:M_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_M,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if M_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                M_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0].M_K_read_cycle_count
            + l2_tiles[0, 0, 0].K_N_read_cycle_count
        ) # 读取第一个矩阵的第一个tile和第二个矩阵的第一个tile

        previous_m = 0 # 先前l2tiles中元素的index
        previous_n = 0 # 先前l2tiles中元素的index
        previous_k = 0 # 先前l2tiles中元素的index

        for m, n, k in self.generate_tile_loops(
            ceil(M / l2_tile_M),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if m == 0 and n == 0 and k == 0:
                continue

            l2_tile = l2_tiles[m, n, k]
            previous_l2_tile = l2_tiles[previous_m, previous_n, previous_k]

            # current tile read latency
            if m == previous_m and k == previous_k:
                current_tile_read_cycle_count = l2_tile.K_N_read_cycle_count
            elif n == previous_n and k == previous_k:
                current_tile_read_cycle_count = l2_tile.M_K_read_cycle_count
            else:
                current_tile_read_cycle_count = (
                    l2_tile.M_K_read_cycle_count + l2_tile.K_N_read_cycle_count
                )
            if k > 0 and not (m == previous_m and n == previous_n): # not后确保只在切换输出矩阵mn tile的位置时才输出结果，把mn tile中间数据暂存于主存，mn不切换时只是累加。k>0要求只有非第一次计算该mn块时才从主存中读取中间数据
                current_tile_read_cycle_count += l2_tile.M_N_read_cycle_count
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if previous_k > 0:  # previous_k>0要求在非第一轮时要对mn块与主存中的中间数据进行按元素求和、读、写（三个部分的cycle）。原文是if k>0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if m == previous_m and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_write_cycle_count

            # read current tile, compute previous tile, write previous tile
            if mapping.is_l2_double_buffering:  # pipelined
                total_cycle_count += (
                    max(
                        current_tile_read_cycle_count, previous_tile_compute_cycle_count
                    )
                    + previous_tile_write_cycle_count
                )
            else:  # non-pipelined
                total_cycle_count += (
                    current_tile_read_cycle_count
                    + previous_tile_compute_cycle_count
                    + previous_tile_write_cycle_count
                )

            previous_m = m
            previous_n = n
            previous_k = k

        # compute and write last tile
        total_cycle_count += (
            l2_tiles[-1, -1, -1].M_N_write_cycle_count
            + l2_tiles[-1, -1, -1].compute_cycle_count
        )

        if previous_k > 0: # 如果最后一个tile的k维不是0，说明还有一轮k维的累加没有进行，这时需要把累加的结果写回内存。收尾工作
            total_cycle_count += ceil(l2_tiles[-1, -1, -1].K_reduction_cycle_count)

        return total_cycle_count #+ ceil(
        # pcb_module.io_module.latency * 2 * pcb_module.compute_module.clock_freq
        # )

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            pcb_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L2 tile: {M} {N} {K}')
            # L2 tile统计IO与计算cycle
            self.M = M
            self.N = N
            self.K = K
            self.K_reduction_cycle_count = ceil(
                M * N / pcb_module.compute_module.total_vector_flops_per_cycle
            ) # K规约也是完成于l1，因此此处没有l2与主存的io时间
            self.K_reduction_io_count = 2 * M * N * data_type.word_size
            self.M_K_read_cycle_count = self.simulate_l2_tile_read_cycle_count(
                M, K, data_type, pcb_module, mapping.matmul_type
            ) # QK阶段读，SV不读
            self.K_N_read_cycle_count = self.simulate_l2_tile_io_cycle_count(
                K, N, data_type, pcb_module, mapping.matmul_type
            ) # QK和SV都读，所以此处用simulate_l2_tile_io_cycle_count而不是simulate_l2_tile_read_cycle_count
            self.M_N_read_cycle_count = 0 # SV和QK都不读，因为MN tile 不应该从主存中读取，只能从l1中读，但是此处计算的是l2与主存的io，因此为0
            self.M_N_write_cycle_count = self.simulate_l2_tile_write_cycle_count(
                M, N, data_type, pcb_module, mapping.matmul_type
            ) # QK不写，SV写
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, K, data_type, mapping, pcb_module, look_up_table
            )

        def simulate_l2_tile_read_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            chiplet_module: Device,
            matmul_type: MatmulType,
        ): # l2 io是指片外内存（dram/hbm）到片内共享缓存（sram）之间的io
            if matmul_type == "SV":
                return 0
            return self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, chiplet_module, matmul_type
            )
        
        def simulate_l2_tile_write_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            chiplet_module: Device,
            matmul_type: MatmulType,
        ): # l2 io是指片外内存（dram/hbm）到片内共享缓存（sram）之间的io
            if matmul_type == "QK":
                return 0
            return self.simulate_l2_tile_io_cycle_count(
                M, N, data_type, chiplet_module, matmul_type
            )

        def simulate_l2_tile_io_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            chiplet_module: Device,
            matmul_type: MatmulType,
        ): # l2 io是指片外内存（dram/hbm）到片内共享缓存（sram）之间的io
            if matmul_type not in ("QK", "SV"):
                raise ValueError(f"Unsupported matmul_type: {matmul_type}")
            return ceil(
                M
                * N
                * data_type.word_size
                / (
                    chiplet_module.io_module.bandwidth
                    / chiplet_module.compute_module.clock_freq
                )
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ) -> int:
            # 在L2 tile内再做L1切分
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N
            l1_tile_K = mapping.l1_tile_K

            M_l1_t = M // l1_tile_M
            N_l1_t = N // l1_tile_N
            K_l1_t = K // l1_tile_K
            M_remain = M % l1_tile_M
            N_remain = N % l1_tile_N
            K_remain = K % l1_tile_K

            l1_tiles = np.empty(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N), ceil(K / l1_tile_K)],
                dtype=Matmul.L1TileSimulator,
            )
            if M_l1_t * N_l1_t * K_l1_t != 0:
                l1_tiles[:M_l1_t, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain != 0:
                l1_tiles[-1, :N_l1_t, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain != 0:
                l1_tiles[:M_l1_t, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if K_remain != 0:
                l1_tiles[:M_l1_t, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain != 0:
                l1_tiles[-1, -1, :K_l1_t] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    l1_tile_K,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * K_remain != 0:
                l1_tiles[-1, :N_l1_t, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    l1_tile_N,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if N_remain * K_remain != 0:
                l1_tiles[:M_l1_t, -1, -1] = Matmul.L1TileSimulator(
                    l1_tile_M,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )
            if M_remain * N_remain * K_remain != 0:
                l1_tiles[-1, -1, -1] = Matmul.L1TileSimulator(
                    M_remain,
                    N_remain,
                    K_remain,
                    data_type,
                    mapping,
                    chiplet_module,
                    look_up_table,
                )

            M_K_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=int
            ) # “查表矩阵”，记录每个 (m_tile, k_tile) 位置对应的 MK 子块元素数量
            M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
            if M_remain > 0:
                M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
            if K_remain > 0:
                M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
            if M_remain > 0 and K_remain > 0:
                M_K_tile_size[-1, -1] = M_remain * K_remain

            K_N_tile_size = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=int
            ) # “查表矩阵”，记录每个 (k_tile, n_tile) 位置对应的 KN 子块元素数量
            K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
            if K_remain > 0:
                K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
            if N_remain > 0:
                K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
            if K_remain > 0 and N_remain > 0:
                K_N_tile_size[-1, -1] = K_remain * N_remain

            M_N_tile_size = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=int
            ) # “查表矩阵”，记录每个 (m_tile, n_tile) 位置对应的 MN 子块元素数量
            M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
            if M_remain > 0:
                M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
            if N_remain > 0:
                M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
            if M_remain > 0 and N_remain > 0:
                M_N_tile_size[-1, -1] = M_remain * N_remain

            total_cycle_count = 0
            previous_batch_Read_M_K = np.zeros(
                [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
            ) # “状态矩阵”，记录上一个batch中每个 (m_tile, k_tile) 位置的 MK 子块是否被读取过，bool值节省内存并且方便后续计算
            previous_batch_Read_K_N = np.zeros(
                [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
            ) # “状态矩阵”，记录上一个batch中每个 (k_tile, n_tile) 位置的 KN 子块是否被读取过
            previous_batch_Read_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            ) # “状态矩阵”，记录上一个batch中每个 (m_tile, n_tile) 位置的 MN 子块是否被读取过
            previous_batch_Write_M_N = np.zeros(
                [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
            ) # “状态矩阵”，记录上一个batch中每个 (m_tile, n_tile) 位置的 MN 子块是否被写入过
            previous_batch_compute_cycle_count = 0
            active_l1_tile_list = [] # 当前一批并行执行的 L1 tile 队列
            for m, n, k in Matmul.generate_tile_loops(
                ceil(M / l1_tile_M),
                ceil(N / l1_tile_N),
                ceil(K / l1_tile_K),
                mapping.l1_loop_order,
            ): # 按 l1_loop_order 生成 L1 tile 执行顺序，并把 tile 按批（最多 core_count 个）打包做流水记账
                active_l1_tile_list.append((m, n, k, l1_tiles[m, n, k]))
                if (
                    m == ceil(M / l1_tile_M) - 1
                    and n == ceil(N / l1_tile_N) - 1
                    and k == ceil(K / l1_tile_K) - 1
                ): # 最后一个tile，结束当前batch的模拟
                    pass
                elif (
                    len(active_l1_tile_list) < chiplet_module.compute_module.core_count
                ): # 如果当前batch的活跃tile数量还没有达到核心数量上限，继续增加tile
                    continue

                assert (
                    len(active_l1_tile_list) <= chiplet_module.compute_module.core_count
                ) # 活跃tile数量不应该超过核心数量上限
                current_batch_Read_M_K = np.zeros(
                    [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
                ) # “状态矩阵”，记录当前batch中每个 (m_tile, k_tile) 位置的 MK 子块是否被读取过
                current_batch_Read_K_N = np.zeros(
                    [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
                ) # “状态矩阵”，记录当前batch中每个 (k_tile, n_tile) 位置的 KN 子块是否被读取过
                current_batch_Read_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                ) # “状态矩阵”，记录当前batch中每个 (m_tile, n_tile) 位置的 MN 子块是否被读取过
                current_batch_Write_M_N = np.zeros(
                    [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
                ) # “状态矩阵”，记录当前batch中每个 (m_tile, n_tile) 位置的 MN 子块是否被写入过

                current_batch_compute_cycle_count = 0
                for i in range(len(active_l1_tile_list)):
                    temp_m, temp_n, temp_k, temp_l1_tile = active_l1_tile_list[i]
                    current_batch_Read_M_K[temp_m, temp_k] = 1
                    current_batch_Read_K_N[temp_k, temp_n] = 1
                    current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
                    current_batch_Write_M_N[temp_m, temp_n] = 1
                    temp_l1_tile_compute_cycle_count = temp_l1_tile.compute_cycle_count # 当前tile的计算cycle count
                    if temp_k > 0:
                        temp_l1_tile_compute_cycle_count += ceil(
                            temp_l1_tile.M
                            * temp_l1_tile.N
                            / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
                        ) # 如果k维不是0，说明不是第一轮累加，需要加上把部分和写回寄存器文件再读出来进行下一轮累加的cycle count
                    current_batch_compute_cycle_count = max(
                        current_batch_compute_cycle_count,
                        temp_l1_tile_compute_cycle_count,
                    ) # 当前batch的计算cycle count取决于活跃tile中计算cycle count最多的那个tile，因为它们是并行执行的

                # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
                # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
                current_batch_M_K_read_count = np.sum(
                    (current_batch_Read_M_K * (~previous_batch_Read_M_K))
                    * M_K_tile_size
                )
                current_batch_K_N_read_count = np.sum(
                    (current_batch_Read_K_N * (~previous_batch_Read_K_N))
                    * K_N_tile_size
                )
                current_batch_M_N_read_count = np.sum(
                    (
                        current_batch_Read_M_N
                        * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
                    )
                    * M_N_tile_size
                )
                if mapping.matmul_type == "QK":
                    previous_batch_M_N_write_count = 0 # Qk不需要写MN
                elif mapping.matmul_type == "SV":
                    previous_batch_M_N_write_count = np.sum(
                        (previous_batch_Write_M_N * (~current_batch_Read_M_N))
                        * M_N_tile_size
                    ) # SV需要写MN
                else:
                    raise ValueError(f"Unsupported matmul_type: {mapping.matmul_type}")

                # read current batch while compute and write previous batch. 先统计读写元素量，再按字节宽度和 L2 带宽折算成 IO 周期，供后面的流水重叠公式使用
                if mapping.matmul_type == "QK":
                    current_batch_read_count = (
                        current_batch_M_K_read_count + current_batch_K_N_read_count
                    ) # QK只读MK和KN
                elif mapping.matmul_type == "SV":
                    current_batch_read_count = current_batch_K_N_read_count # SV只读KN
                else:
                    raise ValueError(
                        f"Unsupported matmul_type: {mapping.matmul_type}"
                    )
                current_batch_read_cycle_count = ceil(
                    current_batch_read_count
                    * chiplet_module.compute_module.core.systolic_array.input_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                ) # l2 bandwith指l2到l1之间的带宽
                prvious_batch_write_cycle_count = ceil(
                    previous_batch_M_N_write_count
                    * chiplet_module.compute_module.core.systolic_array.output_word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                ) # l2 bandwith指l2到l1之间的带宽

                total_cycle_count += (
                    max(
                        current_batch_read_cycle_count,
                        previous_batch_compute_cycle_count,
                    )
                    + prvious_batch_write_cycle_count
                )

                previous_batch_compute_cycle_count = current_batch_compute_cycle_count
                previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
                previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
                previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
                previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

                active_l1_tile_list = []

            # last batch's compute and write. 最后一批的计算和写回通常无法和下一批的读取重叠，所以单独算cycle count
            if mapping.matmul_type == "QK":
                last_batch_write_cycle_count = 0 # QK不需要写
            elif mapping.matmul_type == "SV":
                last_batch_write_cycle_count = ceil(
                    np.sum(previous_batch_Write_M_N * M_N_tile_size)
                    * data_type.word_size
                    / chiplet_module.compute_module.l2_bandwidth_per_cycle
                ) # Sv需要写
            else:
                raise ValueError(f"Unsupported matmul_type: {mapping.matmul_type}")
            total_cycle_count += (
                previous_batch_compute_cycle_count + last_batch_write_cycle_count
            )

            return total_cycle_count

    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # print(f'L1 tile: {M} {N} {K}')
            self.M = M
            self.N = N
            self.K = K
            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, K, data_type, mapping, chiplet_module, look_up_table
            )

        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            K: int,
            data_type: DataType,
            mapping: "Matmul.Mapping",
            chiplet_module: Device,
            look_up_table: pd.DataFrame,
        ):
            # L1 tile内核: 阵列计算+K归约
            assert (
                M * K + K * N + M * N
                <= chiplet_module.compute_module.core.SRAM_size
                // data_type.word_size
                // 2
            )

            M_tiling_factor = mapping.l0_M_tiling_factor
            N_tiling_factor = mapping.l0_N_tiling_factor
            K_tiling_factor = mapping.l0_K_tiling_factor
            assert (
                M_tiling_factor * K_tiling_factor * N_tiling_factor
                <= chiplet_module.compute_module.core.systolic_array_count
            ) # 这里的约束是为了保证在L1 tile的计算过程中，能够把一个tile完全映射到systolic array上进行计算，避免tile内部还需要进行分块和调度，从而增加计算复杂度和cycle count

            compute_cycle_count = ceil(
                Matmul.simulate_systolic_array_cycle_count(
                    look_up_table,
                    ceil(M / M_tiling_factor),
                    ceil(N / N_tiling_factor),
                    ceil(K / K_tiling_factor),
                    chiplet_module.compute_module.core.systolic_array.array_height,
                    chiplet_module.compute_module.core.systolic_array.array_width,
                    chiplet_module.compute_module.core.systolic_array.mac_per_cycle,
                    mapping.dataflow,
                )
                + (K_tiling_factor - 1)
                * M
                * N
                / chiplet_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            ) # 第一部分是计算tile内部的乘加操作需要的cycle count，第二部分是计算tile内部k维累加时需要的cycle count，k维每增加一轮累加，就需要把当前的部分和写回寄存器文件，再读出来和下一轮的乘加结果进行累加，这部分的cycle count可以近似看成是M*N个元素的累加操作，每个元素的累加操作可以看成是一个vector reduce操作，需要的cycle count取决于vector unit的规格

            return compute_cycle_count

    @staticmethod
    def simulate_systolic_array_cycle_count(
        look_up_table: pd.DataFrame,
        M,
        N,
        K,
        array_height,
        array_width,
        mac_per_clock,
        dataflow="os",
    ):
        # print(f'start: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        assert M * N * K * array_height * array_width * mac_per_clock != 0
        # 大尺寸先用利用率近似
        if M >= array_height and N >= array_width:
            if (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 128
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.99
                )
            elif (
                M * N * K / array_height / array_width / max(array_height, array_width)
                >= 64
            ):
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / 0.98
                )
        elif M >= array_height and N < array_width:
            if K * M / array_height / max(array_height, array_width) >= 64:
                util_rate = N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        elif M < array_height and N >= array_width:
            if K * N / array_width / max(array_height, array_width) >= 64:
                util_rate = M / array_height / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        else:
            assert M < array_height and N < array_width
            if K / max(array_height, array_width) >= 64:
                util_rate = M / array_height * N / array_width / 0.98
                return ceil(
                    M * N * K / array_height / array_width / mac_per_clock / util_rate
                )
        # print('start look up table')
        try:
            cycle_count = look_up_table.loc[
                (M, N, K, array_height, array_width, dataflow), "cycle_count"
            ].item()
        except KeyError:
            try:
                cycle_count = look_up_table.loc[
                    (N, M, K, array_height, array_width, dataflow), "cycle_count"
                ].item()
            except KeyError:
                # print('not found in look up table')
                # 查表未命中则调用ScaleSim
                config = f"./systolic_array_model/temp/systolic_array_{os.getpid()}.cfg"
                with open(config, "w") as f:
                    f.writelines("[general]\n")
                    f.writelines("run_name = systolic_array\n\n")
                    f.writelines("[architecture_presets]\n")
                    f.writelines("ArrayHeight:    " + str(array_height) + "\n")
                    f.writelines("ArrayWidth:     " + str(array_width) + "\n")
                    f.writelines("IfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("FilterSramSzkB:   " + str(1024) + "\n")
                    f.writelines("OfmapSramSzkB:    " + str(1024) + "\n")
                    f.writelines("IfmapOffset:    0\n")
                    f.writelines("FilterOffset:   10000000\n")
                    f.writelines("OfmapOffset:    20000000\n")
                    f.writelines("Dataflow : " + dataflow + "\n")
                    f.writelines("Bandwidth : " + "100" + "\n")
                    f.writelines("MemoryBanks: 1\n\n")
                    f.writelines("[run_presets]\n")
                    f.writelines("InterfaceBandwidth: CALC\n")

                topology = f"./systolic_array_model/temp/matmul_{os.getpid()}.csv"
                with open(topology, "w") as f:
                    f.writelines("Layer, M, N, K\n")
                    f.writelines(f"matmul1, {M}, {N}, {K},\n")

                logpath = f"./systolic_array_model/temp/"
                s = scalesim(
                    save_disk_space=True,
                    verbose=False,
                    config=config,
                    topology=topology,
                    input_type_gemm=True,
                )
                s.run_scale(top_path=logpath)

                cycle_count = s.runner.single_layer_sim_object_list[0].total_cycles
                util_rate = s.runner.single_layer_sim_object_list[0].overall_util
                with open(
                    f"./systolic_array_model/look_up_table_{array_height}_{array_width}.csv",
                    "a",
                ) as f:
                    f.writelines(
                        f"{M},{N},{K},{array_height},{array_width},{dataflow},{cycle_count},{util_rate:.3f}\n"
                    )
                look_up_table.loc[(M, N, K, array_height, array_width, dataflow), :] = [
                    cycle_count,
                    util_rate,
                ]
                if len(look_up_table) % 10 == 0:
                    look_up_table.sort_index(inplace=True)
        # if (
        #     dataflow == "os"
        # ):  # scalesim assumes collecting output is not on critical path in os
        #     cycle_count += min(array_height, array_width, M, N)
        # if True:
        #     print(f"{M}x{N}x{K}x{array_height}x{array_width}x{dataflow}: {cycle_count}")
        # new_table = look_up_table[~look_up_table.index.duplicated(keep='first')]
        # if look_up_table.shape[0]-new_table.shape[0]>=1:
        #     print(look_up_table)
        #     print(look_up_table.duplicated(keep=False))
        #     exit()
        # print(f'end: {M} {N} {K} {array_height} {array_width} {mac_per_clock} {dataflow}')
        # assert isinstance(cycle_count, float), f"cycle_count: {cycle_count}"
        # 将阵列周期换算为核心周期
        return ceil(cycle_count / mac_per_clock)

class Softmax(Operator):
    def __init__(self, M: int, N: int, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.M = M
        self.N = N
        self.output_shape = [self.M, self.N]
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.data_type
        )

    def print_latency(self):
        print(f"{self.output_shape}, {self.latency_on_gpu*1e6}us")

    class ComputationalGraph:
        def __init__(self, M: int, N: int, data_type: DataType):
            self.M = M
            self.N = N
            self.data_type = data_type

    class Mapping:
        def __init__(
            self,
            l2_tile_M: int,
            l2_tile_N: int,
            is_l2_double_buffering: bool,
            l1_tile_M: int,
            l1_tile_N: int,
            is_l1_double_buffering: bool = False,
        ):
            self.l2_tile_M = l2_tile_M
            self.l2_tile_N = l2_tile_N
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_M = l1_tile_M
            self.l1_tile_N = l1_tile_N
            self.is_l1_double_buffering = is_l1_double_buffering

        def display(self):
            print("-" * 20)
            print(
                f"l2_tile_M: {self.l2_tile_M}, is_l2_double_buffering: {self.is_l2_double_buffering}, l1_tile_M: {self.l1_tile_M}, l1_tile_N: {self.l1_tile_N}, is_l1_double_buffering: {self.is_l1_double_buffering}"
            )
    
    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        self.computational_graph.data_type = pcb_module.compute_module.core.vector_unit.data_type
        min_cycle_count = float("inf")
        best_mapping = None
        # 将输入张量映射为 M * N 的二维矩阵，其中每行都做softmax，一共做M次softmax
        M = self.computational_graph.M
        N = self.computational_graph.N
        data_type = self.computational_graph.data_type
        l2_tile_N = N
        l2_tile_M = (
            pcb_module.compute_module.l2_size // (l2_tile_N * data_type.word_size)
        ) # l2能放下的矩阵的行数，即单次缓存l2可以做的softmax的次数
        l2_tile_M = max(1, min(l2_tile_M, M))
        is_l2_double_buffering = False
        for l1_N_tiling_factor in [1, 2, 4, 8, 16, 32]:
            l1_tile_N = ceil(l2_tile_N / l1_N_tiling_factor)
            for l1_tile_M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                for is_l1_double_buffering in [True, False]:
                    if is_l1_double_buffering:
                        if (
                            l1_tile_M * l1_tile_N * data_type.word_size
                            > pcb_module.compute_module.core.SRAM_size // 2
                        ):
                            continue
                    else:
                        if (
                            l1_tile_M * l1_tile_N * data_type.word_size
                            > pcb_module.compute_module.core.SRAM_size
                        ):
                            continue
                    mapping = self.Mapping(
                        l2_tile_M,
                        l2_tile_N,
                        is_l2_double_buffering,
                        l1_tile_M,
                        l1_tile_N,
                        is_l1_double_buffering,
                    )
                    cycle_count = self.simulate(
                        self.computational_graph, mapping, pcb_module
                    )
                    if cycle_count < min_cycle_count:
                        min_cycle_count = cycle_count
                        best_mapping = mapping
        self.best_mapping = best_mapping
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.best_cycle_count

    def simulate(
        self,
        computational_graph: ComputationalGraph,
        mapping: Mapping,
        pcb_module: Device,
    ) -> int:
        M = computational_graph.M
        N = computational_graph.N
        data_type = computational_graph.data_type
        l2_tile_M = mapping.l2_tile_M

        if mapping.is_l2_double_buffering:
            assert (
                l2_tile_M * N * data_type.word_size * 2
                <= pcb_module.compute_module.l2_size
            )
        else:
            assert (
                l2_tile_M * N * data_type.word_size <= pcb_module.compute_module.l2_size
            )

        M_l2_t = M // l2_tile_M
        M_remain = M % l2_tile_M

        l2_tiles = np.empty([ceil(M / l2_tile_M)], dtype=self.L2TileSimulator) # l2_tiles用来存放每个l2 tile的模拟结果的数组

        if M_l2_t != 0:
            l2_tiles[:M_l2_t] = self.L2TileSimulator(
                l2_tile_M,
                N,
                data_type,
                mapping,
                pcb_module,
            )
        if M_remain != 0:
            l2_tiles[-1] = self.L2TileSimulator(
                M_remain,
                N,
                data_type,
                mapping,
                pcb_module,
            )

        total_cycle_count = 0
        l2_tile_count = ceil(M / l2_tile_M)
        for m in range(l2_tile_count):
            total_cycle_count += l2_tiles[m].compute_cycle_count # 只有计算没有io
        return ceil(total_cycle_count)

    class L2TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ): # 注意，此处的M，N是指这个L2 tile的shape，而不是整个输入的shape，这里的M，N只是变量名与之前的computational_graph.M、computational_graph.N重复而已
            self.M = M
            self.N = N
            self.compute_cycle_count = self.simulate_l2_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )

        def simulate_l2_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            l1_tile_M = mapping.l1_tile_M
            l1_tile_N = mapping.l1_tile_N

            l1_tile = Softmax.L1TileSimulator(
                l1_tile_M,
                l1_tile_N,
                data_type,
                mapping,
                pcb_module,
            )
            l1_tile_count = ceil(M / l1_tile_M) * ceil(N / l1_tile_N)
            l1_tile_cycle_count = l1_tile.compute_cycle_count  # 只有计算没有io
            reduction_round_count = ceil(log2(ceil(N / l1_tile_N)))
            total_cycle_count = ceil(
                l1_tile_count / pcb_module.compute_module.core_count
            ) * (
                l1_tile_cycle_count
                + reduction_round_count * l1_tile.reduction_cycle_count
            )
            return total_cycle_count


    class L1TileSimulator:
        def __init__(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ): # 注意，与上面相同，此处的M，N是指这个L1 tile的shape
            self.M = M
            self.N = N
            self.flops_per_exp = (
                pcb_module.compute_module.core.vector_unit.flops_per_exp
            )

            self.compute_cycle_count = self.simulate_l1_tile_compute_cycle_count(
                M, N, data_type, mapping, pcb_module
            )

            self.reduction_cycle_count = (
                M
                * N
                * (self.flops_per_exp + 2)
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )


        def simulate_l1_tile_compute_cycle_count(
            self,
            M: int,
            N: int,
            data_type: DataType,
            mapping: "Softmax.Mapping",
            pcb_module: Device,
        ):
            # online softmax
            total_flop_count = M * N * (self.flops_per_exp * 3 + 7) # 经验公式，粗略估计
            return ceil(
                total_flop_count
                / pcb_module.compute_module.core.vector_unit.total_vector_flops_per_cycle
            )

def main() -> int:
    from hardware_model.device import device_dict

    M, K, N = 128, 256, 512
    data_type = data_type_dict["fp16"]
    pcb_module = device_dict["A100_80GB_fp16"]

    matmul = Matmul(M, K, N, data_type, "QK")
    assert matmul.output_shape == [M, N], f"unexpected output shape: {matmul.output_shape}"
    assert matmul.flop_count == 2 * M * K * N
    assert matmul.io_count == M * K + K * N + M * N

    simulate_cycle_count = matmul.compile_and_simulate(pcb_module)

    print("Matmul smoke test passed")
    print(f"M={M}, K={K}, N={N}, dtype={data_type.name}")
    print(f"output_shape={matmul.output_shape}")
    print(f"flops={matmul.flop_count}, io_count={matmul.io_count}")
    print(f"simulate_cycle_count={simulate_cycle_count}")
    print(f"simulate_latency={matmul.best_latency:.6e}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
