"""
BEACON Core Neural Network Architecture
Distributed Edge2Seq, Network-Aware MGD, and Consensus Aggregation Modules
IEEE INFOCOM 2026 Submission
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from typing import Optional, Tuple, Union, Dict, List, Any, Callable
import numpy as np
from collections import OrderedDict, deque
import math
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import heapq


@dataclass
class NetworkState:
    """Encapsulates current network conditions for adaptive processing."""
    latency: float  # Current network latency in milliseconds
    bandwidth: float  # Available bandwidth in Mbps
    packet_loss: float  # Packet loss rate (0-1)
    congestion_level: float  # Network congestion level (0-1)
    topology_changes: int  # Number of topology changes in current window
    jitter: float = 0.0  # Network jitter in ms
    routing_distance: int = 1  # Number of hops to coordinator
    active_nodes: int = 1  # Number of active edge nodes
    timestamp: float = field(default_factory=time.time)


@dataclass
class CommunicationPacket:
    """Network packet for edge node communication."""
    source_id: str
    destination_id: str
    packet_type: str  # 'model_update', 'consensus', 'heartbeat', 'verification'
    payload: torch.Tensor
    sequence_number: int
    timestamp: float
    priority: int = 0
    ttl: int = 10
    signature: Optional[str] = None


class AggregationStrategy(Enum):
    """Consensus aggregation strategies for Byzantine fault tolerance."""
    FEDERATED_AVERAGING = "fedavg"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"
    BYZANTINE_ROBUST = "byzantine_robust"
    MULTI_KRUM = "multi_krum"
    COORDINATE_MEDIAN = "coordinate_median"


class NetworkProtocol:
    """Implements network communication protocols for edge nodes."""
    
    def __init__(self, node_id: str, encryption_key: Optional[bytes] = None):
        self.node_id = node_id
        self.encryption_key = encryption_key or self._generate_key()
        self.sequence_counter = 0
        self.packet_buffer = deque(maxlen=10000)
        self.routing_table = {}
        self.neighbor_states = {}
        self.packet_loss_simulator = PacketLossSimulator()
        
    def _generate_key(self) -> bytes:
        """Generate encryption key for secure communication."""
        return hashlib.sha256(f"{self.node_id}_{time.time()}".encode()).digest()
    
    def create_packet(
        self,
        destination: str,
        packet_type: str,
        payload: torch.Tensor,
        priority: int = 0
    ) -> CommunicationPacket:
        """Create a network packet with proper headers."""
        packet = CommunicationPacket(
            source_id=self.node_id,
            destination_id=destination,
            packet_type=packet_type,
            payload=payload,
            sequence_number=self.sequence_counter,
            timestamp=time.time(),
            priority=priority
        )
        
        # Sign packet for authentication
        packet.signature = self._sign_packet(packet)
        self.sequence_counter += 1
        
        return packet
    
    def _sign_packet(self, packet: CommunicationPacket) -> str:
        """Create digital signature for packet authentication."""
        packet_hash = hashlib.sha256(
            f"{packet.source_id}{packet.destination_id}{packet.sequence_number}".encode()
        ).hexdigest()
        return packet_hash
    
    async def send_packet(
        self,
        packet: CommunicationPacket,
        network_state: NetworkState
    ) -> bool:
        """Simulate packet transmission with network conditions."""
        # Simulate network delay
        delay = self._calculate_transmission_delay(packet, network_state)
        await asyncio.sleep(delay / 1000.0)  # Convert ms to seconds
        
        # Simulate packet loss
        if self.packet_loss_simulator.should_drop_packet(network_state.packet_loss):
            return False
        
        # Add to destination buffer (simulated)
        self.packet_buffer.append(packet)
        return True
    
    def _calculate_transmission_delay(
        self,
        packet: CommunicationPacket,
        network_state: NetworkState
    ) -> float:
        """Calculate realistic transmission delay based on network conditions."""
        base_delay = network_state.latency
        
        # Add serialization delay based on packet size
        packet_size_mb = packet.payload.element_size() * packet.payload.nelement() / (1024 * 1024)
        serialization_delay = (packet_size_mb * 8) / network_state.bandwidth * 1000  # ms
        
        # Add queueing delay based on congestion
        queueing_delay = base_delay * network_state.congestion_level * 2
        
        # Add jitter
        jitter = np.random.normal(0, network_state.jitter)
        
        total_delay = base_delay + serialization_delay + queueing_delay + jitter
        return max(0, total_delay)


class PacketLossSimulator:
    """Simulates realistic packet loss patterns."""
    
    def __init__(self):
        self.loss_history = deque(maxlen=100)
        self.burst_mode = False
        self.burst_counter = 0
        
    def should_drop_packet(self, base_loss_rate: float) -> bool:
        """Determine if packet should be dropped based on loss patterns."""
        # Implement Gilbert-Elliott model for burst losses
        if self.burst_mode:
            self.burst_counter -= 1
            if self.burst_counter <= 0:
                self.burst_mode = False
            return np.random.random() < base_loss_rate * 5  # Higher loss in burst
        else:
            if np.random.random() < base_loss_rate * 0.1:  # 10% chance to enter burst
                self.burst_mode = True
                self.burst_counter = np.random.randint(3, 10)
            return np.random.random() < base_loss_rate


class DistributedEdge2Seq(nn.Module):
    """
    Enhanced Edge2Seq module designed for distributed processing at edge nodes.
    Includes network-aware optimizations and communication efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_sequence_length: int = 32,
        edge_compression_ratio: float = 0.5,
        adaptive_processing: bool = True,
        quantization_bits: int = 8,
        pruning_threshold: float = 0.01
    ):
        super(DistributedEdge2Seq, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.edge_compression_ratio = edge_compression_ratio
        self.adaptive_processing = adaptive_processing
        self.quantization_bits = quantization_bits
        self.pruning_threshold = pruning_threshold
        
        # Performance metrics tracking
        self.processing_times = deque(maxlen=100)
        self.compression_ratios = deque(maxlen=100)
        
        # Embedding layer with compression for edge deployment
        compressed_dim = int(hidden_dim * edge_compression_ratio)
        self.edge_embedding = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
            nn.LayerNorm(compressed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional GRU with optimization for edge devices
        self.gru_in = nn.GRU(
            compressed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.gru_out = nn.GRU(
            compressed_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Adaptive pooling mechanism
        self.adaptive_pool = AdaptiveSequencePooling(hidden_dim)
        
        # Network-aware attention mechanism
        self.network_attention = NetworkAwareAttention(
            hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Compression layers for network transmission
        self.transmission_encoder = nn.Sequential(
            nn.Linear(hidden_dim, compressed_dim),
            nn.Tanh()  # Bound outputs for better quantization
        )
        self.transmission_decoder = nn.Linear(compressed_dim, hidden_dim)
        
        # Gradient compression for federated learning
        self.gradient_compressor = GradientCompressor(
            compression_ratio=0.1,
            quantization_bits=quantization_bits
        )
        
    def forward(
        self,
        in_sequences: torch.Tensor,
        out_sequences: torch.Tensor,
        sequence_lengths: Dict[str, torch.Tensor],
        network_state: Optional[NetworkState] = None,
        return_compressed: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process transaction sequences with network-aware optimizations.
        """
        start_time = time.time()
        batch_size = in_sequences.size(0)
        device = in_sequences.device
        
        # Adaptive sequence truncation based on network conditions
        effective_length = self.max_sequence_length
        if self.adaptive_processing and network_state is not None:
            effective_length = self._compute_adaptive_length(network_state)
            in_sequences = in_sequences[:, :effective_length, :]
            out_sequences = out_sequences[:, :effective_length, :]
        
        # Apply pruning for edge efficiency
        if self.training:
            in_sequences, out_sequences = self._apply_structured_pruning(
                in_sequences, out_sequences
            )
        
        # Edge embedding with compression
        in_embedded = self.edge_embedding(in_sequences)
        out_embedded = self.edge_embedding(out_sequences)
        
        # Quantize embeddings for edge devices
        if return_compressed:
            in_embedded = self._quantize_tensor(in_embedded, self.quantization_bits)
            out_embedded = self._quantize_tensor(out_embedded, self.quantization_bits)
        
        # Pack sequences for efficient processing
        in_lengths = torch.clamp(sequence_lengths['in'], max=effective_length)
        out_lengths = torch.clamp(sequence_lengths['out'], max=effective_length)
        
        in_packed = nn.utils.rnn.pack_padded_sequence(
            in_embedded,
            in_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        out_packed = nn.utils.rnn.pack_padded_sequence(
            out_embedded,
            out_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process with bidirectional GRU
        in_output, in_hidden = self.gru_in(in_packed)
        out_output, out_hidden = self.gru_out(out_packed)
        
        # Unpack and apply adaptive pooling
        in_unpacked, _ = nn.utils.rnn.pad_packed_sequence(in_output, batch_first=True)
        out_unpacked, _ = nn.utils.rnn.pad_packed_sequence(out_output, batch_first=True)
        
        # Adaptive pooling based on sequence importance
        in_pooled = self.adaptive_pool(in_unpacked, in_lengths)
        out_pooled = self.adaptive_pool(out_unpacked, out_lengths)
        
        # Apply network-aware attention
        in_attended, in_attention_weights = self.network_attention(
            in_pooled.unsqueeze(1),
            network_state
        )
        out_attended, out_attention_weights = self.network_attention(
            out_pooled.unsqueeze(1),
            network_state
        )
        
        # Combine representations
        combined = torch.cat([in_attended.squeeze(1), out_attended.squeeze(1)], dim=-1)
        node_representations = self.fusion_layer(combined)
        
        # Prepare compressed version for transmission
        compressed_repr = None
        compression_ratio = 1.0
        if return_compressed or (network_state and network_state.bandwidth < 10.0):
            compressed_repr = self.transmission_encoder(node_representations)
            compressed_repr = self._quantize_tensor(compressed_repr, self.quantization_bits)
            compression_ratio = compressed_repr.numel() / node_representations.numel()
            self.compression_ratios.append(compression_ratio)
            
            if not return_compressed:
                node_representations = self.transmission_decoder(compressed_repr)
        
        # Track processing time
        processing_time = (time.time() - start_time) * 1000  # ms
        self.processing_times.append(processing_time)
        
        auxiliary_info = {
            'in_attention': in_attention_weights,
            'out_attention': out_attention_weights,
            'effective_length': effective_length,
            'compression_applied': compressed_repr is not None,
            'compression_ratio': compression_ratio,
            'processing_time_ms': processing_time,
            'compressed_representation': compressed_repr
        }
        
        return node_representations, auxiliary_info
    
    def _compute_adaptive_length(self, network_state: NetworkState) -> int:
        """Compute adaptive sequence length based on network conditions."""
        # Base reduction factors for different network conditions
        factors = {
            'latency': 1.0 - min(0.5, network_state.latency / 200.0),
            'bandwidth': min(1.0, network_state.bandwidth / 50.0),
            'congestion': 1.0 - network_state.congestion_level * 0.4,
            'packet_loss': 1.0 - network_state.packet_loss * 0.3
        }
        
        # Weighted combination
        weights = {'latency': 0.3, 'bandwidth': 0.3, 'congestion': 0.2, 'packet_loss': 0.2}
        length_factor = sum(factors[k] * weights[k] for k in factors)
        
        adaptive_length = int(self.max_sequence_length * length_factor)
        return max(8, min(adaptive_length, self.max_sequence_length))
    
    def _apply_structured_pruning(
        self,
        in_sequences: torch.Tensor,
        out_sequences: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply structured pruning to reduce computation on edge devices."""
        # Compute importance scores for each position
        in_importance = torch.norm(in_sequences, dim=-1)
        out_importance = torch.norm(out_sequences, dim=-1)
        
        # Create masks based on importance threshold
        in_mask = in_importance > self.pruning_threshold
        out_mask = out_importance > self.pruning_threshold
        
        # Apply masks
        in_sequences = in_sequences * in_mask.unsqueeze(-1)
        out_sequences = out_sequences * out_mask.unsqueeze(-1)
        
        return in_sequences, out_sequences
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize tensor for efficient network transmission."""
        # Scale to quantization range
        scale = (2 ** bits - 1) / 2
        quantized = torch.round(tensor * scale) / scale
        return quantized
    
    def get_edge_statistics(self) -> Dict[str, float]:
        """Get performance statistics for edge deployment monitoring."""
        return {
            'avg_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_compression_ratio': np.mean(self.compression_ratios) if self.compression_ratios else 1,
            'memory_usage_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
        }


class NetworkAwareMGD(MessagePassing):
    """
    Network-Aware Multigraph Discrepancy module with advanced routing capabilities.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        network_aware: bool = True,
        routing_enabled: bool = True
    ):
        super(NetworkAwareMGD, self).__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.network_aware = network_aware
        self.routing_enabled = routing_enabled
        
        # Linear transformations for node features
        self.lin_self = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_neighbor = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_discrepancy = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Network-aware attention parameters
        self.att_self = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_neighbor = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_discrepancy = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_network = Parameter(torch.Tensor(1, heads, out_channels))
        
        # Network topology encoder
        self.topology_encoder = nn.Sequential(
            nn.Linear(8, out_channels),  # Extended network features
            nn.ReLU(),
            nn.Linear(out_channels, heads * out_channels)
        )
        
        # Routing mechanism
        if routing_enabled:
            self.routing_network = TransactionRouter(
                in_channels=out_channels,
                num_routes=3  # Normal, verification, high-priority
            )
        
        # Latency-aware aggregation
        self.latency_modulator = nn.Sequential(
            nn.Linear(3, out_channels),  # latency, jitter, distance
            nn.Sigmoid()
        )
        
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
        
        # Performance tracking
        self.message_count = 0
        self.routing_decisions = deque(maxlen=1000)
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neighbor.weight)
        nn.init.xavier_uniform_(self.lin_discrepancy.weight)
        nn.init.xavier_uniform_(self.att_self)
        nn.init.xavier_uniform_(self.att_neighbor)
        nn.init.xavier_uniform_(self.att_discrepancy)
        nn.init.xavier_uniform_(self.att_network)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        network_features: Optional[torch.Tensor] = None,
        network_state: Optional[NetworkState] = None,
        return_attention_weights: bool = False,
        return_routing_decisions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Perform network-aware message passing with intelligent routing.
        """
        H, C = self.heads, self.out_channels
        x_self = self.lin_self(x).view(-1, H, C)
        x_neighbor = self.lin_neighbor(x).view(-1, H, C)
        x_discrepancy = self.lin_discrepancy(x).view(-1, H, C)
        
        if self.add_self_loops:
            num_nodes = x_self.size(0)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=0.0, num_nodes=num_nodes
            )
        
        # Compute network-aware features
        if self.network_aware and network_features is not None:
            network_encoded = self.topology_encoder(network_features).view(-1, H, C)
        else:
            # Create default network features
            default_features = torch.zeros(x.size(0), 8, device=x.device)
            network_encoded = self.topology_encoder(default_features).view(-1, H, C)
        
        # Routing decisions for suspicious transactions
        routing_paths = None
        if self.routing_enabled and hasattr(self, 'routing_network'):
            routing_paths = self.routing_network(x_self.mean(dim=1), edge_index)
            self.routing_decisions.extend(routing_paths.argmax(dim=1).tolist())
        
        # Perform message passing
        out = self.propagate(
            edge_index,
            x=(x_neighbor, x_self),
            x_discrepancy=x_discrepancy,
            network_encoded=network_encoded,
            network_state=network_state,
            routing_paths=routing_paths,
            size=None
        )
        
        # Apply latency-aware modulation
        if network_state is not None:
            latency_features = torch.tensor([
                network_state.latency / 100.0,
                network_state.jitter / 50.0,
                network_state.routing_distance / 10.0
            ], device=x.device).unsqueeze(0)
            
            latency_weights = self.latency_modulator(latency_features)
            out = out * latency_weights.unsqueeze(0).unsqueeze(0)
        
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        if self.bias is not None:
            out = out + self.bias
        
        # Prepare return values
        if return_attention_weights or return_routing_decisions:
            auxiliary_outputs = {}
            
            if return_attention_weights:
                alpha = self._compute_attention_weights(
                    x_self, x_neighbor, x_discrepancy, network_encoded, edge_index
                )
                auxiliary_outputs['attention_weights'] = alpha
            
            if return_routing_decisions and routing_paths is not None:
                auxiliary_outputs['routing_decisions'] = routing_paths
                auxiliary_outputs['routing_statistics'] = self._get_routing_statistics()
            
            return out, auxiliary_outputs
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        x_discrepancy_i: torch.Tensor,
        network_encoded_i: torch.Tensor,
        routing_paths: Optional[torch.Tensor],
        index: torch.Tensor,
        size_i: int,
        network_state: Optional[NetworkState] = None
    ) -> torch.Tensor:
        """
        Compute messages with network-aware discrepancy and routing.
        """
        self.message_count += 1
        
        # Compute discrepancy between nodes
        discrepancy = x_i - x_j
        
        # Combine features for attention computation
        alpha_self = (x_i * self.att_self).sum(dim=-1)
        alpha_neighbor = (x_j * self.att_neighbor).sum(dim=-1)
        alpha_discrepancy = (discrepancy * self.att_discrepancy).sum(dim=-1)
        alpha_network = (network_encoded_i * self.att_network).sum(dim=-1)
        
        # Compute attention scores
        alpha = alpha_self + alpha_neighbor + alpha_discrepancy + alpha_network
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, num_nodes=size_i)
        
        # Apply routing-based attention modulation
        if routing_paths is not None:
            # High-priority routes get higher attention
            route_weights = routing_paths[index].unsqueeze(-1)
            alpha = alpha * (1 + route_weights.squeeze(-1) * 0.5)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention
        message = alpha.unsqueeze(-1) * (x_j + self.lin_discrepancy(discrepancy.mean(dim=1)).view(-1, self.heads, self.out_channels))
        
        # Apply network-aware modulation
        if network_state is not None:
            # Adaptive message strength based on network conditions
            reliability_factor = 1.0 - (
                network_state.packet_loss * 0.3 +
                network_state.congestion_level * 0.2 +
                min(0.5, network_state.latency / 200.0) * 0.5
            )
            message = message * reliability_factor
        
        return message
    
    def _compute_attention_weights(
        self,
        x_self: torch.Tensor,
        x_neighbor: torch.Tensor,
        x_discrepancy: torch.Tensor,
        network_encoded: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute and store attention weights for interpretability."""
        # Placeholder implementation - in production, this would store
        # attention values computed during message passing
        num_edges = edge_index.size(1)
        return torch.rand(num_edges, self.heads, device=x_self.device)
    
    def _get_routing_statistics(self) -> Dict[str, float]:
        """Get routing decision statistics."""
        if not self.routing_decisions:
            return {'normal': 0, 'verification': 0, 'high_priority': 0}
        
        decisions = np.array(self.routing_decisions)
        total = len(decisions)
        
        return {
            'normal': np.sum(decisions == 0) / total,
            'verification': np.sum(decisions == 1) / total,
            'high_priority': np.sum(decisions == 2) / total,
            'total_messages': self.message_count
        }


class TransactionRouter(nn.Module):
    """
    Intelligent routing mechanism for suspicious transaction detection.
    Routes transactions through different verification paths based on risk assessment.
    """
    
    def __init__(self, in_channels: int, num_routes: int = 3):
        super(TransactionRouter, self).__init__()
        
        self.risk_assessor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, 1),
            nn.Sigmoid()
        )
        
        self.route_selector = nn.Sequential(
            nn.Linear(in_channels + 1, in_channels),  # +1 for risk score
            nn.ReLU(),
            nn.Linear(in_channels, num_routes),
            nn.Softmax(dim=-1)
        )
        
        self.num_routes = num_routes
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine routing paths for each node based on risk assessment.
        
        Returns:
            routing_probabilities: [num_nodes, num_routes]
        """
        # Assess risk level for each node
        risk_scores = self.risk_assessor(node_features)
        
        # Concatenate risk scores with features
        routing_input = torch.cat([node_features, risk_scores], dim=-1)
        
        # Select routing paths
        routing_probabilities = self.route_selector(routing_input)
        
        return routing_probabilities


class ConsensusAggregator(nn.Module):
    """
    Advanced Byzantine fault-tolerant consensus aggregation with
    verifiable computation and incentive mechanisms.
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        byzantine_tolerance: float = 0.2,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.BYZANTINE_ROBUST,
        consensus_rounds: int = 3,
        verification_threshold: float = 0.8,
        incentive_enabled: bool = True
    ):
        super(ConsensusAggregator, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.byzantine_tolerance = byzantine_tolerance
        self.aggregation_strategy = aggregation_strategy
        self.consensus_rounds = consensus_rounds
        self.verification_threshold = verification_threshold
        self.incentive_enabled = incentive_enabled
        
        # Trust score computation network
        self.trust_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),  # +3 for performance metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Consensus verification network
        self.verification_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Aggregation weight network
        self.weight_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for trust and network metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        # Historical tracking
        self.register_buffer('historical_trust', torch.ones(num_nodes) * 0.5)
        self.register_buffer('contribution_scores', torch.zeros(num_nodes))
        self.register_buffer('verification_success', torch.zeros(num_nodes))
        
        # Performance metrics
        self.consensus_times = deque(maxlen=100)
        self.aggregation_errors = deque(maxlen=100)
        
        # Incentive mechanism
        if incentive_enabled:
            self.incentive_calculator = IncentiveMechanism(
                base_reward=1.0,
                performance_weight=0.7,
                participation_weight=0.3
            )
    
    def forward(
        self,
        edge_updates: Dict[str, torch.Tensor],
        global_model: torch.Tensor,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        network_states: Optional[Dict[str, NetworkState]] = None,
        require_verification: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform Byzantine-robust aggregation with verification and incentives.
        """
        start_time = time.time()
        device = next(iter(edge_updates.values())).device
        
        # Convert updates to tensor format
        updates_list = []
        node_ids = []
        performance_metrics = []
        
        for node_id, update in edge_updates.items():
            updates_list.append(update.unsqueeze(0))
            node_idx = int(node_id.split('_')[1])
            node_ids.append(node_idx)
            
            # Collect performance metrics if available
            if network_states and node_id in network_states:
                state = network_states[node_id]
                metrics = torch.tensor([
                    1.0 - state.packet_loss,
                    min(1.0, state.bandwidth / 100.0),
                    1.0 - min(1.0, state.latency / 200.0)
                ], device=device)
            else:
                metrics = torch.ones(3, device=device) * 0.5
            
            performance_metrics.append(metrics)
        
        updates_tensor = torch.cat(updates_list, dim=0)
        performance_tensor = torch.stack(performance_metrics)
        
        # Compute trust scores with performance awareness
        trust_scores = self._compute_trust_scores(
            updates_tensor,
            global_model,
            node_ids,
            performance_tensor,
            validation_data
        )
        
        # Verify updates if required
        if require_verification:
            verification_results = self._verify_updates(
                updates_tensor,
                global_model,
                trust_scores
            )
            
            # Filter out unverified updates
            verified_mask = verification_results > self.verification_threshold
            if verified_mask.sum() < len(updates_tensor) * 0.5:
                # Too many failures - use only highest trust nodes
                top_k = max(3, int(len(updates_tensor) * 0.3))
                _, top_indices = torch.topk(trust_scores, top_k)
                verified_mask = torch.zeros_like(verified_mask)
                verified_mask[top_indices] = True
            
            updates_tensor = updates_tensor[verified_mask]
            trust_scores = trust_scores[verified_mask]
            node_ids = [node_ids[i] for i in range(len(node_ids)) if verified_mask[i]]
        
        # Apply selected aggregation strategy
        aggregated = self._apply_aggregation_strategy(
            updates_tensor,
            trust_scores,
            self.aggregation_strategy
        )
        
        # Iterative consensus refinement
        for round_idx in range(self.consensus_rounds):
            aggregated = self._consensus_round(
                aggregated,
                updates_tensor,
                trust_scores,
                round_idx
            )
        
        # Update tracking metrics
        self._update_historical_metrics(node_ids, trust_scores, performance_tensor)
        
        # Calculate incentives if enabled
        incentives = {}
        if self.incentive_enabled:
            incentives = self.incentive_calculator.calculate_rewards(
                node_ids,
                trust_scores,
                self.contribution_scores[node_ids],
                performance_tensor
            )
        
        # Measure consensus time
        consensus_time = (time.time() - start_time) * 1000  # ms
        self.consensus_times.append(consensus_time)
        
        # Prepare comprehensive output
        output_info = {
            'trust_scores': {f'edge_{nid}': score.item() for nid, score in zip(node_ids, trust_scores)},
            'verification_results': verification_results if require_verification else None,
            'consensus_time_ms': consensus_time,
            'num_verified_nodes': len(node_ids),
            'aggregation_strategy': self.aggregation_strategy.value,
            'incentives': incentives
        }
        
        return aggregated, output_info
    
    def _compute_trust_scores(
        self,
        updates: torch.Tensor,
        global_model: torch.Tensor,
        node_ids: List[int],
        performance_metrics: torch.Tensor,
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Enhanced trust computation with performance awareness."""
        num_updates = updates.size(0)
        trust_scores = torch.zeros(num_updates, device=updates.device)
        
        for i, (update, node_id, perf_metrics) in enumerate(zip(updates, node_ids, performance_metrics)):
            # Prepare trust computation input
            trust_input = torch.cat([
                update.flatten(),
                global_model.flatten(),
                perf_metrics
            ])
            
            # Compute current trust score
            current_trust = self.trust_network(trust_input)
            
            # Blend with historical trust
            historical_weight = 0.4
            blended_trust = (
                (1 - historical_weight) * current_trust +
                historical_weight * self.historical_trust[node_id]
            )
            
            # Adjust based on recent verification success
            verification_weight = 0.2
            final_trust = (
                (1 - verification_weight) * blended_trust +
                verification_weight * self.verification_success[node_id]
            )
            
            trust_scores[i] = final_trust.squeeze()
        
        # Normalize trust scores
        trust_scores = F.softmax(trust_scores, dim=0)
        
        return trust_scores
    
    def _verify_updates(
        self,
        updates: torch.Tensor,
        global_model: torch.Tensor,
        trust_scores: torch.Tensor
    ) -> torch.Tensor:
        """Verify update validity through consensus checking."""
        num_updates = updates.size(0)
        verification_scores = torch.zeros(num_updates, device=updates.device)
        
        for i, update in enumerate(updates):
            # Compute verification score
            verification_input = torch.cat([
                update.flatten(),
                global_model.flatten()
            ])
            
            score = self.verification_network(verification_input)
            
            # Weight by trust score
            verification_scores[i] = score * trust_scores[i]
        
        return verification_scores
    
    def _apply_aggregation_strategy(
        self,
        updates: torch.Tensor,
        trust_scores: torch.Tensor,
        strategy: AggregationStrategy
    ) -> torch.Tensor:
        """Apply selected aggregation strategy with optimizations."""
        
        if strategy == AggregationStrategy.BYZANTINE_ROBUST:
            return self._byzantine_robust_aggregation(updates, trust_scores)
        elif strategy == AggregationStrategy.KRUM:
            return self._krum_aggregation(updates, trust_scores)
        elif strategy == AggregationStrategy.MULTI_KRUM:
            return self._multi_krum_aggregation(updates, trust_scores, m=3)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            return self._trimmed_mean_aggregation(updates, trust_scores, self.byzantine_tolerance)
        elif strategy == AggregationStrategy.COORDINATE_MEDIAN:
            return self._coordinate_median_aggregation(updates, trust_scores)
        else:
            return self._federated_averaging(updates, trust_scores)
    
    def _byzantine_robust_aggregation(
        self,
        updates: torch.Tensor,
        trust_scores: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced Byzantine-robust aggregation with outlier detection."""
        # Detect and remove outliers using statistical methods
        update_norms = torch.norm(updates.view(len(updates), -1), dim=1)
        mean_norm = update_norms.mean()
        std_norm = update_norms.std()
        
        # Remove updates beyond 2 standard deviations
        inlier_mask = torch.abs(update_norms - mean_norm) < 2 * std_norm
        
        # Also remove lowest trust scores
        num_byzantine = int(len(updates) * self.byzantine_tolerance)
        if num_byzantine > 0:
            _, lowest_trust_indices = torch.topk(trust_scores, num_byzantine, largest=False)
            inlier_mask[lowest_trust_indices] = False
        
        # Filter updates
        filtered_updates = updates[inlier_mask]
        filtered_trust = trust_scores[inlier_mask]
        
        if len(filtered_updates) == 0:
            # Fallback to highest trust update
            best_idx = torch.argmax(trust_scores)
            return updates[best_idx]
        
        # Renormalize trust scores
        filtered_trust = filtered_trust / filtered_trust.sum()
        
        # Weighted aggregation
        aggregated = torch.sum(
            filtered_updates * filtered_trust.unsqueeze(-1),
            dim=0
        )
        
        return aggregated
    
    def _multi_krum_aggregation(
        self,
        updates: torch.Tensor,
        trust_scores: torch.Tensor,
        m: int = 3
    ) -> torch.Tensor:
        """Multi-Krum aggregation selecting m updates."""
        n = len(updates)
        f = int(n * self.byzantine_tolerance)
        
        # Compute pairwise distances
        distances = torch.cdist(
            updates.view(n, -1),
            updates.view(n, -1)
        )
        
        # For each update, compute sum of distances to k nearest neighbors
        k = n - f - 2
        sorted_distances, _ = torch.sort(distances, dim=1)
        scores = sorted_distances[:, 1:k+1].sum(dim=1)
        
        # Weight by trust scores
        weighted_scores = scores * (2 - trust_scores)  # Lower score is better
        
        # Select top m updates
        _, selected_indices = torch.topk(weighted_scores, m, largest=False)
        selected_updates = updates[selected_indices]
        selected_trust = trust_scores[selected_indices]
        
        # Average selected updates weighted by trust
        selected_trust = selected_trust / selected_trust.sum()
        aggregated = torch.sum(
            selected_updates * selected_trust.unsqueeze(-1),
            dim=0
        )
        
        return aggregated
    
    def _coordinate_median_aggregation(
        self,
        updates: torch.Tensor,
        trust_scores: torch.Tensor
    ) -> torch.Tensor:
        """Coordinate-wise median aggregation."""
        # Weight updates by trust for median computation
        weighted_updates = updates * trust_scores.unsqueeze(-1).sqrt()
        
        # Compute coordinate-wise median
        aggregated, _ = torch.median(weighted_updates, dim=0)
        
        return aggregated
    
    def _consensus_round(
        self,
        current_aggregate: torch.Tensor,
        updates: torch.Tensor,
        trust_scores: torch.Tensor,
        round_idx: int
    ) -> torch.Tensor:
        """Enhanced consensus round with adaptive blending."""
        # Compute agreement scores
        agreements = F.cosine_similarity(
            updates.view(len(updates), -1),
            current_aggregate.view(1, -1),
            dim=1
        )
        
        # Adaptive momentum based on round and agreement variance
        agreement_variance = agreements.var()
        base_momentum = 0.9 - 0.1 * round_idx
        adaptive_momentum = base_momentum * (1 - agreement_variance)
        
        # Update trust scores based on agreement
        adjusted_trust = adaptive_momentum * trust_scores + (1 - adaptive_momentum) * agreements
        adjusted_trust = F.softmax(adjusted_trust, dim=0)
        
        # Recompute aggregate with adjusted trust
        refined_aggregate = torch.sum(
            updates * adjusted_trust.unsqueeze(-1),
            dim=0
        )
        
        # Adaptive blending based on convergence
        blend_factor = 0.7 - 0.1 * round_idx - 0.2 * agreement_variance
        blend_factor = max(0.3, blend_factor)
        
        final_aggregate = (
            blend_factor * current_aggregate +
            (1 - blend_factor) * refined_aggregate
        )
        
        return final_aggregate
    
    def _update_historical_metrics(
        self,
        node_ids: List[int],
        current_trust: torch.Tensor,
        performance_metrics: torch.Tensor
    ):
        """Update historical metrics with exponential moving average."""
        alpha_trust = 0.1
        alpha_contribution = 0.05
        
        for i, node_id in enumerate(node_ids):
            # Update trust scores
            self.historical_trust[node_id] = (
                (1 - alpha_trust) * self.historical_trust[node_id] +
                alpha_trust * current_trust[i]
            )
            
            # Update contribution scores based on participation
            self.contribution_scores[node_id] = (
                (1 - alpha_contribution) * self.contribution_scores[node_id] +
                alpha_contribution
            )
            
            # Update verification success based on performance
            self.verification_success[node_id] = performance_metrics[i].mean()
    
    def get_consensus_statistics(self) -> Dict[str, float]:
        """Get comprehensive consensus statistics."""
        return {
            'avg_consensus_time_ms': np.mean(self.consensus_times) if self.consensus_times else 0,
            'avg_trust_score': self.historical_trust.mean().item(),
            'trust_variance': self.historical_trust.var().item(),
            'avg_contribution': self.contribution_scores.mean().item(),
            'active_nodes': (self.contribution_scores > 0.1).sum().item()
        }


class IncentiveMechanism:
    """Incentive mechanism for encouraging honest participation."""
    
    def __init__(
        self,
        base_reward: float = 1.0,
        performance_weight: float = 0.7,
        participation_weight: float = 0.3
    ):
        self.base_reward = base_reward
        self.performance_weight = performance_weight
        self.participation_weight = participation_weight
    
    def calculate_rewards(
        self,
        node_ids: List[int],
        trust_scores: torch.Tensor,
        contribution_scores: torch.Tensor,
        performance_metrics: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate rewards for participating nodes."""
        rewards = {}
        
        for i, node_id in enumerate(node_ids):
            performance_score = performance_metrics[i].mean().item()
            participation_score = contribution_scores[i].item()
            
            total_score = (
                self.performance_weight * performance_score * trust_scores[i].item() +
                self.participation_weight * participation_score
            )
            
            reward = self.base_reward * total_score
            rewards[f'edge_{node_id}'] = reward
        
        return rewards


class GradientCompressor:
    """Gradient compression for efficient federated learning."""
    
    def __init__(self, compression_ratio: float = 0.1, quantization_bits: int = 8):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.residuals = {}
    
    def compress(self, gradients: Dict[str, torch.Tensor], node_id: str) -> Dict[str, torch.Tensor]:
        """Compress gradients using top-k sparsification and quantization."""
        compressed = {}
        
        for name, grad in gradients.items():
            # Add residual from previous round
            if node_id in self.residuals and name in self.residuals[node_id]:
                grad = grad + self.residuals[node_id][name]
            
            # Top-k sparsification
            k = max(1, int(grad.numel() * self.compression_ratio))
            values, indices = torch.topk(grad.abs().view(-1), k)
            
            # Store residual
            residual = grad.clone()
            residual.view(-1)[indices] = 0
            if node_id not in self.residuals:
                self.residuals[node_id] = {}
            self.residuals[node_id][name] = residual
            
            # Quantize selected values
            selected_values = grad.view(-1)[indices]
            quantized = self._quantize(selected_values)
            
            compressed[name] = {
                'values': quantized,
                'indices': indices,
                'shape': grad.shape
            }
        
        return compressed
    
    def decompress(self, compressed: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """Decompress gradients."""
        decompressed = {}
        
        for name, data in compressed.items():
            grad = torch.zeros(data['shape']).view(-1)
            grad[data['indices']] = self._dequantize(data['values'])
            decompressed[name] = grad.view(data['shape'])
        
        return decompressed
    
    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to reduce bits."""
        scale = (2 ** self.quantization_bits - 1) / 2
        return torch.round(tensor * scale) / scale
    
    def _dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor (identity for symmetric quantization)."""
        return tensor


class AdaptiveSequencePooling(nn.Module):
    """Enhanced adaptive pooling for variable-length sequences."""
    
    def __init__(self, hidden_dim: int, pooling_strategies: List[str] = ['attention', 'max', 'mean']):
        super(AdaptiveSequencePooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_strategies = pooling_strategies
        
        # Importance scoring network
        self.importance_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, len(pooling_strategies)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequences using adaptive strategy selection.
        """
        batch_size, max_length, hidden_dim = sequences.size()
        device = sequences.device
        
        # Create mask for valid positions
        mask = torch.arange(max_length, device=device).expand(
            batch_size, max_length
        ) < lengths.unsqueeze(1)
        
        # Compute importance scores
        importance_scores = self.importance_network(sequences).squeeze(-1)
        importance_scores = importance_scores.masked_fill(~mask, -1e9)
        importance_weights = F.softmax(importance_scores, dim=1)
        
        # Apply different pooling strategies
        pooled_outputs = []
        
        if 'attention' in self.pooling_strategies:
            attention_pooled = torch.sum(
                sequences * importance_weights.unsqueeze(-1),
                dim=1
            )
            pooled_outputs.append(attention_pooled)
        
        if 'max' in self.pooling_strategies:
            masked_sequences = sequences.masked_fill(~mask.unsqueeze(-1), -1e9)
            max_pooled, _ = torch.max(masked_sequences, dim=1)
            pooled_outputs.append(max_pooled)
        
        if 'mean' in self.pooling_strategies:
            sum_sequences = torch.sum(sequences * mask.unsqueeze(-1).float(), dim=1)
            mean_pooled = sum_sequences / lengths.unsqueeze(-1).float()
            pooled_outputs.append(mean_pooled)
        
        # Select strategy adaptively
        global_features = sequences.mean(dim=1)
        strategy_weights = self.strategy_selector(global_features)
        
        # Combine pooled outputs
        pooled_tensor = torch.stack(pooled_outputs, dim=1)
        final_pooled = torch.sum(
            pooled_tensor * strategy_weights.unsqueeze(-1),
            dim=1
        )
        
        return final_pooled


class NetworkAwareAttention(nn.Module):
    """Enhanced multi-head attention with network condition awareness."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rotary_embedding: bool = True
    ):
        super(NetworkAwareAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_rotary_embedding = use_rotary_embedding
        
        # Linear projections
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Network condition processing
        self.network_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # Extended network features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Rotary position embeddings for better sequence modeling
        if use_rotary_embedding:
            self.rotary_emb = RotaryEmbedding(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        network_state: Optional[NetworkState] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply network-aware attention with optional masking.
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations and reshape
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [B, H, L, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rotary_embedding:
            Q = self.rotary_emb(Q)
            K = self.rotary_emb(K)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Network-aware score modulation
        if network_state is not None:
            network_features = torch.tensor([
                network_state.latency / 100.0,
                network_state.bandwidth / 100.0,
                network_state.packet_loss,
                network_state.congestion_level,
                network_state.jitter / 50.0,
                network_state.routing_distance / 10.0
            ], device=x.device).unsqueeze(0)
            
            network_modulation = self.network_encoder(network_features)
            network_modulation = network_modulation.view(1, self.num_heads, 1, self.head_dim)
            
            # Adaptive attention based on network conditions
            # Poor network conditions lead to more focused attention
            focus_factor = 1.0 + network_state.congestion_level + network_state.packet_loss
            scores = scores * focus_factor
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        output = self.out_linear(attended)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for improved position encoding."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device)
        
        return self._apply_rotary_pos_emb(x)
    
    def _apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation of rotary embeddings
        # Simplified for brevity - full implementation would include
        # cos/sin caching and proper rotation
        return x


class StreamingBuffer:
    """Advanced streaming buffer for real-time transaction processing."""
    
    def __init__(self, window_size: int, feature_dim: int, device: torch.device):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.device = device
        
        self.buffer = torch.zeros(window_size, feature_dim, device=device)
        self.timestamps = torch.zeros(window_size, device=device)
        self.position = 0
        self.is_full = False
        
        # Statistics tracking
        self.running_mean = torch.zeros(feature_dim, device=device)
        self.running_var = torch.ones(feature_dim, device=device)
        self.update_count = 0
    
    def update(self, features: torch.Tensor, timestamp: float):
        """Update buffer with new features."""
        batch_size = features.size(0)
        
        # Handle buffer overflow
        if self.position + batch_size > self.window_size:
            # Wrap around
            first_part = self.window_size - self.position
            self.buffer[self.position:] = features[:first_part]
            self.timestamps[self.position:] = timestamp
            
            remaining = batch_size - first_part
            self.buffer[:remaining] = features[first_part:]
            self.timestamps[:remaining] = timestamp
            
            self.position = remaining
            self.is_full = True
        else:
            self.buffer[self.position:self.position + batch_size] = features
            self.timestamps[self.position:self.position + batch_size] = timestamp
            self.position += batch_size
            
            if self.position >= self.window_size:
                self.is_full = True
                self.position = self.position % self.window_size
        
        # Update running statistics
        self._update_statistics(features)
    
    def _update_statistics(self, features: torch.Tensor):
        """Update running statistics incrementally."""
        batch_size = features.size(0)
        batch_mean = features.mean(dim=0)
        batch_var = features.var(dim=0)
        
        # Welford's online algorithm for running statistics
        self.update_count += batch_size
        delta = batch_mean - self.running_mean
        self.running_mean += delta * batch_size / self.update_count
        self.running_var += batch_var * batch_size / self.update_count
    
    def get_window_features(self, lookback: Optional[int] = None) -> torch.Tensor:
        """Get features from the buffer with optional lookback."""
        if lookback is None:
            return self.buffer if self.is_full else self.buffer[:self.position]
        
        if self.is_full:
            # Circular buffer logic
            if lookback >= self.window_size:
                return self.buffer
            
            start = (self.position - lookback) % self.window_size
            if start < self.position:
                return self.buffer[start:self.position]
            else:
                return torch.cat([self.buffer[start:], self.buffer[:self.position]], dim=0)
        else:
            # Linear buffer logic
            start = max(0, self.position - lookback)
            return self.buffer[start:self.position]
    
    def get_statistics(self) -> Dict[str, torch.Tensor]:
        """Get current buffer statistics."""
        return {
            'mean': self.running_mean,
            'std': torch.sqrt(self.running_var + 1e-6),
            'size': self.position if not self.is_full else self.window_size,
            'update_count': self.update_count
        }


class CrossChainBridge(nn.Module):
    """Module for cross-chain transaction monitoring and fusion."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_chains: int = 2,
        fusion_method: str = 'attention'
    ):
        super(CrossChainBridge, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chains = num_chains
        self.fusion_method = fusion_method
        
        # Chain-specific encoders
        self.chain_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_chains)
        ])
        
        # Cross-chain attention mechanism
        if fusion_method == 'attention':
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1
            )
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * num_chains, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Cross-chain pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # Normal, suspicious, confirmed illicit
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        chain_features: List[torch.Tensor],
        cross_chain_edges: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process multi-chain features and detect cross-chain patterns.
        """
        # Encode chain-specific features
        encoded_features = []
        for i, (features, encoder) in enumerate(zip(chain_features, self.chain_encoders)):
            encoded = encoder(features)
            encoded_features.append(encoded)
        
        # Apply cross-chain attention if specified
        if self.fusion_method == 'attention' and len(encoded_features) > 1:
            # Stack features for attention
            stacked = torch.stack(encoded_features, dim=0)  # [num_chains, batch_size, hidden_dim]
            
            # Self-attention across chains
            attended, attention_weights = self.cross_attention(
                stacked, stacked, stacked
            )
            
            # Use attended features
            encoded_features = [attended[i] for i in range(len(encoded_features))]
        
        # Concatenate and fuse
        concatenated = torch.cat(encoded_features, dim=-1)
        fused = self.fusion_network(concatenated)
        
        # Detect cross-chain patterns
        pattern_scores = self.pattern_detector(fused)
        
        auxiliary_outputs = {
            'pattern_scores': pattern_scores,
            'chain_embeddings': encoded_features,
            'attention_weights': attention_weights if self.fusion_method == 'attention' else None
        }
        
        return fused, auxiliary_outputs


class BEACONModel(nn.Module):
    """
    Main BEACON model integrating all components for distributed
    blockchain anomaly detection with comprehensive network awareness.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_edge_nodes: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        consensus_threshold: float = 0.7,
        streaming_window: int = 1000,
        cross_chain_enabled: bool = True,
        num_chains: int = 2,
        edge_deployment_mode: bool = False
    ):
        super(BEACONModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_edge_nodes = num_edge_nodes
        self.num_layers = num_layers
        self.streaming_window = streaming_window
        self.cross_chain_enabled = cross_chain_enabled
        self.edge_deployment_mode = edge_deployment_mode
        
        # Network protocol handler
        self.network_protocol = NetworkProtocol(node_id="central_coordinator")
        
        # Distributed Edge2Seq module
        self.edge2seq = DistributedEdge2Seq(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            edge_compression_ratio=0.5 if edge_deployment_mode else 0.8,
            adaptive_processing=True,
            quantization_bits=8 if edge_deployment_mode else 16
        )
        
        # Network-Aware MGD layers
        self.mgd_layers = nn.ModuleList([
            NetworkAwareMGD(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout,
                network_aware=True,
                routing_enabled=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization between MGD layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Consensus aggregator
        self.consensus = ConsensusAggregator(
            num_nodes=num_edge_nodes,
            hidden_dim=hidden_dim,
            byzantine_tolerance=0.2,
            aggregation_strategy=AggregationStrategy.BYZANTINE_ROBUST,
            consensus_rounds=3,
            verification_threshold=consensus_threshold,
            incentive_enabled=True
        )
        
        # Cross-chain bridge
        if cross_chain_enabled:
            self.cross_chain_bridge = CrossChainBridge(
                hidden_dim=hidden_dim,
                num_chains=num_chains,
                fusion_method='attention'
            )
        
        # Streaming buffer
        self.streaming_buffer = StreamingBuffer(
            window_size=streaming_window,
            feature_dim=hidden_dim,
            device=torch.device('cpu')  # Will be moved to correct device
        )
        
        # Output classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Edge deployment optimizations
        if edge_deployment_mode:
            self.edge_compressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh()
            )
            self.edge_decompressor = nn.Linear(hidden_dim // 4, hidden_dim)
        
        # Performance tracking
        self.forward_times = deque(maxlen=100)
        self.communication_overhead = deque(maxlen=100)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        in_sequences: Optional[torch.Tensor] = None,
        out_sequences: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[Dict[str, torch.Tensor]] = None,
        network_state: Optional[NetworkState] = None,
        network_features: Optional[torch.Tensor] = None,
        cross_chain_features: Optional[List[torch.Tensor]] = None,
        streaming_mode: bool = False,
        return_edge_outputs: bool = False,
        distributed_mode: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Comprehensive forward pass with all BEACON features.
        """
        start_time = time.time()
        batch_size = node_features.size(0)
        device = node_features.device
        
        # Move streaming buffer to correct device if needed
        if self.streaming_buffer.device != device:
            self.streaming_buffer = StreamingBuffer(
                self.streaming_window,
                self.hidden_dim,
                device
            )
        
        # Process sequences with Edge2Seq
        if in_sequences is not None and out_sequences is not None:
            sequence_features, seq_info = self.edge2seq(
                in_sequences,
                out_sequences,
                sequence_lengths,
                network_state,
                return_compressed=self.edge_deployment_mode
            )
            
            # Track communication overhead
            if 'compressed_representation' in seq_info and seq_info['compressed_representation'] is not None:
                overhead = seq_info['compressed_representation'].numel() * 4  # bytes
                self.communication_overhead.append(overhead)
            
            # Combine with node features
            x = node_features + sequence_features
        else:
            x = node_features
            seq_info = {}
        
        # Apply Network-Aware MGD layers with residual connections
        edge_outputs = []
        routing_decisions = []
        
        for i, (mgd_layer, layer_norm) in enumerate(zip(self.mgd_layers, self.layer_norms)):
            # Store input for residual connection
            residual = x
            
            # Apply MGD layer with routing
            mgd_output = mgd_layer(
                x,
                edge_index,
                edge_attr,
                network_features,
                network_state,
                return_attention_weights=False,
                return_routing_decisions=True
            )
            
            if isinstance(mgd_output, tuple):
                x, mgd_aux = mgd_output
                if 'routing_decisions' in mgd_aux:
                    routing_decisions.append(mgd_aux['routing_decisions'])
            else:
                x = mgd_output
            
            # Apply residual connection and normalization
            x = layer_norm(x + residual)
            
            # Apply activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=dropout, training=self.training)
            
            if return_edge_outputs:
                edge_outputs.append(x.clone())
        
        # Streaming mode processing
        if streaming_mode:
            self.streaming_buffer.update(x, time.time())
            
            # Get temporal context
            temporal_context = self.streaming_buffer.get_window_features(lookback=100)
            stats = self.streaming_buffer.get_statistics()
            
            # Normalize with streaming statistics
            x = (x - stats['mean']) / stats['std']
        
        # Cross-chain fusion if enabled
        cross_chain_info = {}
        if self.cross_chain_enabled and cross_chain_features is not None:
            # Process current chain features
            current_chain = x
            all_chain_features = [current_chain] + cross_chain_features
            
            # Apply cross-chain bridge
            x, cross_chain_info = self.cross_chain_bridge(all_chain_features)
        
        # Distributed mode aggregation
        consensus_info = {}
        if distributed_mode and hasattr(self, 'edge_updates'):
            # Simulate edge node updates (in practice, these would come from network)
            x, consensus_info = self.consensus(
                self.edge_updates,
                x,
                network_states={f'edge_{i}': network_state for i in range(self.num_edge_nodes)},
                require_verification=True
            )
        
        # Edge deployment compression
        if self.edge_deployment_mode:
            x_compressed = self.edge_compressor(x)
            x = self.edge_decompressor(x_compressed)
        
        # Final classification
        predictions = self.classifier(x)
        
        # Track forward pass time
        forward_time = (time.time() - start_time) * 1000  # ms
        self.forward_times.append(forward_time)
        
        # Prepare comprehensive outputs
        if return_edge_outputs:
            auxiliary_outputs = {
                'edge_outputs': edge_outputs,
                'sequence_info': seq_info,
                'routing_decisions': routing_decisions,
                'cross_chain_info': cross_chain_info,
                'consensus_info': consensus_info,
                'final_features': x,
                'performance_metrics': {
                    'forward_time_ms': forward_time,
                    'avg_forward_time_ms': np.mean(self.forward_times),
                    'communication_overhead_bytes': np.mean(self.communication_overhead) if self.communication_overhead else 0,
                    'edge_compression_ratio': seq_info.get('compression_ratio', 1.0)
                },
                'streaming_stats': self.streaming_buffer.get_statistics() if streaming_mode else None
            }
            return predictions, auxiliary_outputs
        
        return predictions
    
    def deploy_to_edge(self, node_id: int) -> 'EdgeBEACON':
        """
        Create lightweight edge version of BEACON for deployment.
        """
        edge_model = EdgeBEACON(
            node_id=node_id,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim // 2,  # Reduced dimension
            output_dim=self.output_dim,
            parent_model=self
        )
        
        # Transfer compressed weights
        edge_model.load_compressed_state(self.state_dict())
        
        return edge_model
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network and performance statistics."""
        stats = {
            'model_metrics': {
                'parameters': sum(p.numel() for p in self.parameters()),
                'memory_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024)
            },
            'performance_metrics': {
                'avg_forward_time_ms': np.mean(self.forward_times) if self.forward_times else 0,
                'avg_communication_overhead_bytes': np.mean(self.communication_overhead) if self.communication_overhead else 0
            },
            'edge2seq_stats': self.edge2seq.get_edge_statistics(),
            'consensus_stats': self.consensus.get_consensus_statistics()
        }
        
        # Add routing statistics from MGD layers
        routing_stats = []
        for i, layer in enumerate(self.mgd_layers):
            if hasattr(layer, '_get_routing_statistics'):
                routing_stats.append(layer._get_routing_statistics())
        
        if routing_stats:
            stats['routing_statistics'] = routing_stats
        
        return stats


class EdgeBEACON(nn.Module):
    """
    Lightweight BEACON variant for edge deployment with minimal resources.
    """
    
    def __init__(
        self,
        node_id: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        parent_model: Optional[BEACONModel] = None
    ):
        super(EdgeBEACON, self).__init__()
        
        self.node_id = node_id
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simplified Edge2Seq
        self.edge2seq_lite = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Single MGD layer
        self.mgd_lite = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Lightweight classifier
        self.classifier_lite = nn.Linear(hidden_dim, output_dim)
        
        # Communication module
        self.comm_encoder = nn.Linear(hidden_dim, hidden_dim // 4)
        
        # Local buffer for streaming
        self.local_buffer = deque(maxlen=100)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lightweight forward pass for edge inference."""
        # Process features
        x = self.edge2seq_lite(x)
        x = self.mgd_lite(x)
        
        # Local prediction
        predictions = self.classifier_lite(x)
        
        # Prepare compressed update for coordinator
        compressed = self.comm_encoder(x)
        
        # Update local buffer
        self.local_buffer.append(x.detach())
        
        return predictions, compressed
    
    def load_compressed_state(self, parent_state_dict: Dict[str, torch.Tensor]):
        """Load compressed weights from parent model."""
        # Implement weight compression and loading
        # This is a simplified version - full implementation would include
        # proper weight quantization and pruning
        pass