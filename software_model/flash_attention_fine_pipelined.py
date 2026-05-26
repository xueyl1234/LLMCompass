from utils import size
from typing import List, Tuple
from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import Tensor, DataType
from math import ceil, log2, floor
from itertools import permutations
import torch
import time
import statistics
import numpy as np
import pandas as pd
import os
from scalesim.scale_sim import scalesim
import copy

# 注意
# 1: 关于l2 l1双缓冲容量的问题，兴许可以使用更小的容量占用，因为Q用完就没用了
# 2: py默认K和V矩阵都采用了完全相同的分块策略，二者的分块矩阵在l2和l1种完全相同

class FlashAttention(Operator): # 包含计算注意力分数QK^T、Softmax以及把 softmax 后权重乘以 V，得到上下文向量 AV。初始数据应该与计算注意力分数QK^T所需的一致
    def __init__(self, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.input1_shape = None
        self.input2_shape = None
        self.output_shape = None
        self.look_up_table = None
        self.best_mapping = None

    def __call__(self, input1: Tensor, input2: Tensor) -> Tensor:
        # [B, M, K] * [B, K, N] = [B, M, N]
        assert self.data_type == input1.data_type
        assert self.data_type == input2.data_type
        self.input1_shape = input1.shape
        self.input2_shape = input2.shape
        if len(self.input1_shape) >= 3 and len(self.input2_shape) >= 3:
            assert size(self.input1_shape[:-2]) == size(self.input2_shape[:-2])
            self.B = size(self.input1_shape[:-2])
            self.M = self.input1_shape[-2]
            self.K = self.input1_shape[-1]
            assert self.input2_shape[-2] == self.K
            self.N = self.input2_shape[-1]
            self.output_shape = self.input1_shape[:-2] + [self.M, self.N]
        else:
            self.B = 1
            self.M = self.input1_shape[-2]
            self.K = self.input1_shape[-1]
            assert self.input2_shape[-2] == self.K
            self.N = self.input2_shape[-1]
            self.output_shape = [self.M, self.N]
        output = Tensor(self.output_shape, self.data_type)
        self.computational_graph = self.ComputationalGraph(
            self.M, self.N, self.K, self.data_type
        )
        self.flop_count = 2 * self.B * self.M * self.K * self.N
        self.io_count = self.B * (
            self.M * self.K + self.K * self.N + self.M * self.N
        )
        # print(f'{self.M}, {self.N}, {self.K}')
        return output

    def print_latency(self):
        print(
            f"{self.computational_graph.M}, {self.computational_graph.N}, {self.computational_graph.K}, {self.best_latency*1e3:.4f}ms, {self.latency_on_gpu*1e3:.4f}ms, {self.best_latency/self.latency_on_gpu*100:.2f}%",
            flush=True,
        )

    @staticmethod
    def generate_tile_loops(
        loop_B: int, loop_N: int, loop_K1: int, loop_K2: int, loop_order: str
    ):
        tokens = []
        i = 0
        while i < len(loop_order):
            if loop_order.startswith("k1", i):
                tokens.append("k1")
                i += 2
            elif loop_order.startswith("k2", i):
                tokens.append("k2")
                i += 2
            elif loop_order[i] in ("b", "n"):
                tokens.append(loop_order[i])
                i += 1
            else:
                raise ValueError(f"invalid loop_order {loop_order}")

        if len(tokens) != 4 or set(tokens) != {"b", "n", "k1", "k2"}:
            raise ValueError(f"invalid loop_order {loop_order}")

        loop_ranges = {
            "b": range(loop_B),
            "n": range(loop_N),
            "k1": range(loop_K1),
            "k2": range(loop_K2),
        }
        state = {}

        def dfs(depth: int):
            if depth == 4:
                yield state["b"], state["n"], state["k1"], state["k2"]
                return
            dim = tokens[depth]
            for idx in loop_ranges[dim]:
                state[dim] = idx
                yield from dfs(depth + 1)

        yield from dfs(0)

    class ComputationalGraph:
        def __init__(self, B: int, M: int, N: int, K: int, data_type: DataType):
            self.B = B
            self.M = M
            self.N = N
            self.K = K
            self.data_type = data_type

    class Mapping:
        def __init__(
            self,
            M,
            l2_tile_B: int,
            l2_tile_N: int,
            l2_tile_K: int,
            is_l2_double_buffering: bool,
            l1_tile_B: int,
            l1_tile_N: int,
            l1_tile_K: int,
            l2_loop_order: str,
            l1_loop_order: str,
            l0_B_tiling_factor: int,
            l0_N_tiling_factor: int,
            l0_K_tiling_factor: int,
            dataflow: str = "os",
        ):
            self.M = M
            self.l2_tile_B = l2_tile_B
            self.l2_tile_N = l2_tile_N
            self.l2_tile_K = l2_tile_K
            self.is_l2_double_buffering = is_l2_double_buffering
            self.l1_tile_B = l1_tile_B
            self.l1_tile_N = l1_tile_N
            self.l1_tile_K = l1_tile_K
            self.l2_loop_order = l2_loop_order
            self.l1_loop_order = l1_loop_order
            self.l0_B_tiling_factor = l0_B_tiling_factor
            self.l0_N_tiling_factor = l0_N_tiling_factor
            self.l0_K_tiling_factor = l0_K_tiling_factor
            self.dataflow = dataflow


    @staticmethod
    def find_permutations(n): # 为 l0_B_tiling_factor, l0_N_tiling_factor, l0_K_tiling_factor 提供候选分解（把总并行资源 n 按 B/N/K 三个方向分配），n 是 systolic_array_count
        permutations = set()

        for i in range(1, n + 1):
            if n % i == 0:
                for j in range(1, n + 1):
                    if (n // i) % j == 0:
                        k = n // (i * j)
                        permutations.add((i, j, k))

        return list(permutations)

    def compile_and_simulate(
        self,
        pcb_module: Device,
        compile_mode: str = "heuristic-GPU",
    ):        
        # 搜索最优mapping对应最小cycle
        min_cycle_count = 2**63 - 1
        best_mapping = None
        B = 78 # batchsize=78
        M = 8 # GQA 8
        N = 2048 # seq len=2048
        K = 128 # dim head=128

        if compile_mode == "exhaustive":
            # exhaustive: 全参数穷举
            for l2_tile_B_log2 in range(0, ceil(log2(B)) + 1):
                l2_tile_B = 2**l2_tile_B_log2 # l2_tile_B，l2中B的tile size，取2的整数倍次方为了缩减搜索空间。不见得合适
                for l2_tile_N_log2 in range(
                    5, ceil(log2(self.computational_graph.N)) + 1
                ):
                    l2_tile_N = 2**l2_tile_N_log2 # l2_tile_N，l2中N的tile size，取2的整数倍次方为了缩减搜索空间
                    for l2_tile_K_log2 in range(
                        5, ceil(log2(self.computational_graph.K)) + 1
                    ):
                        l2_tile_K = 2**l2_tile_K_log2 # l2_tile_K，l2中M的tile size，取2的整数倍次方为了缩减搜索空
                        working_set_size = l2_tile_B * (
                            l2_tile_N * l2_tile_K * 2
                            + M * l2_tile_K
                            + M * l2_tile_N
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

                        for l1_tile_B_log2 in range(0, l2_tile_B_log2 + 1):
                            l1_tile_B = 2**l1_tile_B_log2 # 缩小搜索空间
                            for l1_tile_N_log2 in range(5, l2_tile_N_log2 + 1):
                                l1_tile_N = 2**l1_tile_N_log2 # 缩小搜索空间
                                for l1_tile_K_log2 in range(5, l2_tile_K_log2 + 1):
                                    l1_tile_K = 2**l1_tile_K_log2 # 缩小搜索空间
                                    if (l1_tile_B * (
                                        M * l1_tile_N
                                        + l1_tile_N * l1_tile_K * 2
                                        + M * l1_tile_K + 2
                                    )> pcb_module.compute_module.core.SRAM_size
                                        // self.data_type.word_size
                                        // 2
                                    ):
                                        continue # l1必须双缓冲，所以将working set size of l1限制在l1大小的1/2
                                    loop_order_candidates = [
                                        "".join(p) for p in permutations(["b", "n", "k1", "k2"])
                                    ]
                                    for l2_loop_order in loop_order_candidates:
                                        for l1_loop_order in loop_order_candidates:
                                            for (
                                                l0_B_tiling_factor,
                                                l0_N_tiling_factor,
                                                l0_K_tiling_factor,
                                            ) in self.find_permutations(
                                                pcb_module.compute_module.core.systolic_array_count
                                            ): # l0 是脉动阵列层级的并行计算划分策略，必须是systolic array count的约数
                                                mapping = self.Mapping(
                                                    M,
                                                    l2_tile_B,
                                                    l2_tile_N,
                                                    l2_tile_K,
                                                    is_l2_double_buffering,
                                                    l1_tile_B,
                                                    l1_tile_N,
                                                    l1_tile_K,
                                                    l2_loop_order,
                                                    l1_loop_order,
                                                    l0_B_tiling_factor,
                                                    l0_N_tiling_factor,
                                                    l0_K_tiling_factor,
                                                )
                                                cycle_count = self.flash_attn_simulate(
                                                    self.computational_graph,
                                                    mapping,
                                                    pcb_module,
                                                )

                                                if cycle_count < min_cycle_count:
                                                    min_cycle_count = cycle_count
                                                    best_mapping = mapping
                                                    # 记录全局最优结果
        self.best_mapping = best_mapping
        # if self.best_mapping is not None:
        #     self.best_mapping.display()
        self.best_cycle_count = min_cycle_count
        self.best_latency = min_cycle_count / pcb_module.compute_module.clock_freq
        self.latency = self.best_latency
        # self.best_mapping.display()
        return self.latency
    
    def flash_attn_simulate(
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
            
        B = computational_graph.B
        M = computational_graph.M
        N = computational_graph.N
        K = computational_graph.K
        data_type = computational_graph.data_type

        l2_tile_B = mapping.l2_tile_B
        l2_tile_N = mapping.l2_tile_N
        l2_tile_K = mapping.l2_tile_K

        if mapping.is_l2_double_buffering:
            assert (l2_tile_B * (l2_tile_N * l2_tile_K * 2 + M * l2_tile_K + M * l2_tile_N)
                <= pcb_module.compute_module.l2_size // self.data_type.word_size // 2
            ) # 容量计算需要重新考虑
        else:
            assert (l2_tile_B * (l2_tile_N * l2_tile_K * 2 + M * l2_tile_K + M * l2_tile_N)
                <= pcb_module.compute_module.l2_size // self.data_type.word_size
            )

        B_l2_t = B // l2_tile_B
        N_l2_t = N // l2_tile_N
        K_l2_t = K // l2_tile_K
        B_remain = B % l2_tile_B
        N_remain = N % l2_tile_N
        K_remain = K % l2_tile_K

        # 按mapping切分L2 tile网格
        l2_tiles = np.empty(
            [ceil(B / l2_tile_B), ceil(N / l2_tile_N), ceil(K / l2_tile_K)],
            dtype=self.L2TileSimulator,
        ) # 这里创建一个空的numpy数组，元素类型是L2TileSimulator实例

        if B_l2_t * N_l2_t * K_l2_t != 0:
            l2_tiles[:B_l2_t, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                l2_tile_B,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            ) # 该切片（数组的子块）中的每个元素都变成一个 L2TileSimulator 实例（同一组参数）
        if B_remain != 0:
            l2_tiles[-1, :N_l2_t, :K_l2_t] = self.L2TileSimulator(
                B_remain,
                l2_tile_N,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain != 0:
            l2_tiles[:B_l2_t, -1, :K_l2_t] = self.L2TileSimulator(
                l2_tile_B,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if K_remain != 0:
            l2_tiles[:B_l2_t, :N_l2_t, -1] = self.L2TileSimulator(
                l2_tile_B,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if B_remain * N_remain != 0:
            l2_tiles[-1, -1, :K_l2_t] = self.L2TileSimulator(
                B_remain,
                N_remain,
                l2_tile_K,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if B_remain * K_remain != 0:
            l2_tiles[-1, :N_l2_t, -1] = self.L2TileSimulator(
                B_remain,
                l2_tile_N,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if N_remain * K_remain != 0:
            l2_tiles[:B_l2_t, -1, -1] = self.L2TileSimulator(
                l2_tile_B,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )
        if B_remain * N_remain * K_remain != 0:
            l2_tiles[-1, -1, -1] = self.L2TileSimulator(
                B_remain,
                N_remain,
                K_remain,
                data_type,
                mapping,
                pcb_module,
                self.look_up_table,
            )

        total_cycle_count = 0
        total_cycle_count += (
            l2_tiles[0, 0, 0].M_K_io_cycle_count + 2 * l2_tiles[0, 0, 0].K_N_io_cycle_count
        ) # 读取第一个矩阵的第一个tile和第二个矩阵的第一个tile

        previous_b = 0 # 先前l2tiles中元素的index
        previous_n = 0 # 先前l2tiles中元素的index
        previous_k1 = 0 # 先前l2tiles中元素的index
        previous_k2 = 0 # 先前l2tiles中元素的index

        for b, n, k1, k2 in self.generate_tile_loops(
            ceil(B / l2_tile_B),
            ceil(N / l2_tile_N),
            ceil(K / l2_tile_K),
            ceil(K / l2_tile_K),
            mapping.l2_loop_order,
        ):
            if b == 0 and n == 0 and k1 == 0 and k2 == 0:
                continue

            l2_tile = l2_tiles[b, n, k1]
            previous_l2_tile = l2_tiles[previous_b, previous_n, previous_k1]

            # current tile read latency
            # Q: MxK1, K: K1xN, V: K2xN
            q_changed = not (b == previous_b and k1 == previous_k1)
            k_changed = not (b == previous_b and n == previous_n and k1 == previous_k1)
            v_changed = not (b == previous_b and n == previous_n and k2 == previous_k2)

            current_tile_read_cycle_count = 0
            if q_changed:
                current_tile_read_cycle_count += l2_tile.M_K_io_cycle_count
            if k_changed:
                current_tile_read_cycle_count += l2_tile.K_N_io_cycle_count
            if v_changed:
                current_tile_read_cycle_count += l2_tile.K_N_io_cycle_count
            if (k1 > 0 or k2 > 0) and not (b == previous_b and n == previous_n): # not后确保只在切换输出矩阵mn tile的位置时才输出结果，把mn tile中间数据暂存于主存，mn不切换时只是累加。k>0要求只有非第一次计算该mn块时才从主存中读取中间数据
                current_tile_read_cycle_count += l2_tile.M_N_io_cycle_count
            # previous tile compute latency
            previous_tile_compute_cycle_count = previous_l2_tile.compute_cycle_count
            if previous_k1 > 0 or previous_k2 > 0:  # previous_k>0要求在非第一轮时要对mn块与主存中的中间数据进行按元素求和、读、写（三个部分的cycle）。原文是if k>0:
                previous_tile_compute_cycle_count += (
                    previous_l2_tile.K_reduction_cycle_count
                )
            # previous tile write latency
            if b == previous_b and n == previous_n:
                previous_tile_write_cycle_count = 0
            else:
                previous_tile_write_cycle_count = previous_l2_tile.M_N_io_cycle_count

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

            previous_b = b
            previous_n = n
            previous_k1 = k1
            previous_k2 = k2

        # compute and write last tile
