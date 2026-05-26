from math import ceil
from typing import Dict

from nandmachine.config.hardware_config import Device
from nandmachine.config.interconnect_config import (
    InterConnectModule,
    MOE_DEFAULT_PARAMS,
    TopologyType,
)


def _word_size_from_weight_bits(weight_bits: int) -> int:
    supported_weight_bits = {8, 16}
    if weight_bits not in supported_weight_bits:
        raise ValueError(
            f"Unsupported weight_bits={weight_bits}, expected one of {sorted(supported_weight_bits)}"
        )
    return weight_bits // 8


ALLREDUCE_DEFAULT_PARAMS: Dict[str, float] = {
    "load_imbalance_factor": 1.0,
    "internal_copy_multiplier": 0.0,
    "gpus_per_node": MOE_DEFAULT_PARAMS["gpus_per_node"],
    "inter_node_bandwidth_per_direction": MOE_DEFAULT_PARAMS[
        "inter_node_bandwidth_per_direction"
    ],
    "inter_node_latency": MOE_DEFAULT_PARAMS["inter_node_latency"],
    "inter_node_header_size": MOE_DEFAULT_PARAMS["inter_node_header_size"],
    "inter_node_max_payload_size": MOE_DEFAULT_PARAMS["inter_node_max_payload_size"],
    "inter_node_link_count_per_device": MOE_DEFAULT_PARAMS[
        "inter_node_link_count_per_device"
    ],
    "inter_node_oversubscription_factor": MOE_DEFAULT_PARAMS[
        "inter_node_oversubscription_factor"
    ],
}


class CommunicationPrimitive:
    def __init__(self, weight_bits: int = 16) -> None:
        self.weight_bits = weight_bits
        self.word_size = _word_size_from_weight_bits(weight_bits)
        # simulation results
        self.latency = None


class AllReduceSimulation(CommunicationPrimitive):
    def __init__(
        self,
        num_gpus: int,
        data_size: int,
        weight_bits: int = 16,
        allreduce_params: Dict[str, float] = ALLREDUCE_DEFAULT_PARAMS,
    ) -> None:
        super().__init__(weight_bits)
        if num_gpus <= 0:
            raise ValueError("num_gpus must be > 0.")
        if data_size < 0:
            raise ValueError("data_size must be >= 0.")
        if data_size % self.word_size != 0:
            raise ValueError(
                f"data_size must be divisible by word_size={self.word_size}, got data_size={data_size}"
            )
        self.num_gpus = int(num_gpus)
        self.data_size = float(data_size)
        self.allreduce_params: Dict[str, float] = dict(ALLREDUCE_DEFAULT_PARAMS)
        self.allreduce_params.update(allreduce_params)
        self.cycle_count = None

    def _single_layer_collective_phase_latency(
        self,
        participant_count: int,
        link_bandwidth_per_direction: float,
        link_latency: float,
        header_size: float,
        max_payload_size: float,
        link_count_per_device: float,
        bytes_per_device: float,
        imbalance_factor: float,
    ) -> float:
        if participant_count <= 1 or bytes_per_device <= 0:
            return 0.0

        bytes_per_peer = bytes_per_device / (participant_count - 1)
        effective_bytes_per_peer = (
            header_size
            + ceil(bytes_per_peer / max_payload_size) * header_size
            + bytes_per_peer
        )
        edge_bandwidth_per_direction = (
            link_bandwidth_per_direction
            * link_count_per_device
            / (participant_count - 1)
        )
        return imbalance_factor * (
            link_latency + effective_bytes_per_peer / edge_bandwidth_per_direction
        )

    def simulate(self, interconnect_module: InterConnectModule) -> float:
        if interconnect_module.device_count != self.num_gpus:
            raise ValueError(
                "interconnect_module.device_count must equal num_gpus, got "
                f"{interconnect_module.device_count} vs {self.num_gpus}"
            )
        if interconnect_module.topology != TopologyType.FC:
            raise NotImplementedError("Dense all-reduce supports FC topology only.")
        if self.num_gpus <= 1 or self.data_size == 0:
            self.latency = 0.0
            return self.latency

        link_bandwidth_per_direction = (
            interconnect_module.link_module.bandwidth_per_direction
        )
        link_latency = interconnect_module.link_module.latency
        header_size = interconnect_module.link_module.header_size
        max_payload_size = interconnect_module.link_module.max_payload_size
        link_count_per_device = interconnect_module.link_count_per_device
        load_imbalance_factor = self.allreduce_params["load_imbalance_factor"]

        # data_size semantics: bytes one rank sends to one peer rank.
        bytes_per_device = self.data_size * (self.num_gpus - 1)

        if self.num_gpus <= 8:
            reduce_scatter_latency = self._single_layer_collective_phase_latency(
                participant_count=self.num_gpus,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
            allgather_latency = self._single_layer_collective_phase_latency(
                participant_count=self.num_gpus,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
        else:
            gpus_per_node = int(self.allreduce_params["gpus_per_node"])
            if gpus_per_node <= 1:
                raise ValueError("gpus_per_node must be > 1 for hierarchical all-reduce.")
            if self.num_gpus % gpus_per_node != 0:
                raise ValueError(
                    "For num_gpus > 8, num_gpus must be divisible by gpus_per_node."
                )
            node_count = self.num_gpus // gpus_per_node

            intra_share = (gpus_per_node - 1) / (self.num_gpus - 1)
            inter_share = (self.num_gpus - gpus_per_node) / (self.num_gpus - 1)
            intra_bytes_per_device = bytes_per_device * intra_share
            inter_bytes_per_device = bytes_per_device * inter_share

            reduce_scatter_intra_latency = self._single_layer_collective_phase_latency(
                participant_count=gpus_per_node,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=intra_bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
            reduce_scatter_inter_latency = self._single_layer_collective_phase_latency(
                participant_count=node_count,
                link_bandwidth_per_direction=self.allreduce_params[
                    "inter_node_bandwidth_per_direction"
                ],
                link_latency=self.allreduce_params["inter_node_latency"],
                header_size=self.allreduce_params["inter_node_header_size"],
                max_payload_size=self.allreduce_params["inter_node_max_payload_size"],
                link_count_per_device=self.allreduce_params[
                    "inter_node_link_count_per_device"
                ],
                bytes_per_device=inter_bytes_per_device,
                imbalance_factor=load_imbalance_factor
                * self.allreduce_params["inter_node_oversubscription_factor"],
            )
            allgather_intra_latency = self._single_layer_collective_phase_latency(
                participant_count=gpus_per_node,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=intra_bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
            allgather_inter_latency = self._single_layer_collective_phase_latency(
                participant_count=node_count,
                link_bandwidth_per_direction=self.allreduce_params[
                    "inter_node_bandwidth_per_direction"
                ],
                link_latency=self.allreduce_params["inter_node_latency"],
                header_size=self.allreduce_params["inter_node_header_size"],
                max_payload_size=self.allreduce_params["inter_node_max_payload_size"],
                link_count_per_device=self.allreduce_params[
                    "inter_node_link_count_per_device"
                ],
                bytes_per_device=inter_bytes_per_device,
                imbalance_factor=load_imbalance_factor
                * self.allreduce_params["inter_node_oversubscription_factor"],
            )

            reduce_scatter_latency = max(
                reduce_scatter_intra_latency, reduce_scatter_inter_latency
            )
            allgather_latency = max(allgather_intra_latency, allgather_inter_latency)

        internal_copy_bytes = self.allreduce_params["internal_copy_multiplier"] * (
            2 * bytes_per_device
        )
        internal_latency = (
            internal_copy_bytes
            / interconnect_module.internal_link_bandwidth_per_direction
        )

        self.latency = reduce_scatter_latency + allgather_latency + internal_latency
        return self.latency

    def compile_and_simulate(
        self,
        pcb_module: Device,
        interconnect_module: InterConnectModule,
        compile_mode: str = "heuristic-GPU",
    ) -> int:
        supported_compile_modes = {
            "exhaustive",
            "heuristic-our-throughput",
            "heuristic-GPU",
        }
        if compile_mode not in supported_compile_modes:
            raise ValueError(
                f"compile_mode {compile_mode} not supported for AllReduceSimulation"
            )

        latency_sec = self.simulate(interconnect_module)
        self.cycle_count = ceil(latency_sec * pcb_module.compute_module.clock_freq)
        return self.cycle_count


class Broadcast:
    def __init__(self):
        self.src = None
        self.tensor = None

    def __call__(self, src: int, tensor: object):
        self.src = src
        self.tensor = tensor
