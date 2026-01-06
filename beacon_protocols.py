"""
BEACON Networking Protocols Module
Adaptive Transaction Routing, Cross-Chain Communication, and Streaming Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict, OrderedDict
import time
import hashlib
import struct
from abc import ABC, abstractmethod
import asyncio
import threading
from queue import PriorityQueue, Queue
import json
import heapq
from concurrent.futures import ThreadPoolExecutor
import socket
import math


@dataclass
class NetworkMetrics:
    """Comprehensive network performance metrics."""
    bandwidth: float  # Mbps
    latency: float  # ms
    packet_loss: float  # percentage
    jitter: float  # ms
    throughput: float  # transactions/second
    congestion_level: float  # 0.0 to 1.0


@dataclass
class Transaction:
    """Encapsulates transaction data with network metadata."""
    tx_id: str
    source: str
    destination: str
    amount: float
    timestamp: float
    chain_id: str
    attributes: torch.Tensor
    risk_score: float = 0.0
    routing_path: List[str] = field(default_factory=list)
    verification_status: str = "pending"
    network_latency: float = 0.0
    priority: int = 0
    retry_count: int = 0
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        return self.priority < other.priority


@dataclass
class RoutingDecision:
    """Represents a routing decision for a transaction."""
    transaction_id: str
    path_type: str  # 'normal', 'verification', 'quarantine'
    path_nodes: List[str]
    confidence: float
    estimated_latency: float
    bandwidth_required: float
    priority_level: int
    alternative_paths: List[List[str]] = field(default_factory=list)
    qos_guarantee: bool = False


class RoutingStrategy(Enum):
    """Available routing strategies for adaptive routing."""
    SHORTEST_PATH = "shortest"
    LOAD_BALANCED = "load_balanced"
    SECURITY_OPTIMIZED = "security"
    LATENCY_OPTIMIZED = "latency"
    HYBRID = "hybrid"
    QOS_AWARE = "qos_aware"


class ChainType(Enum):
    """Supported blockchain types for cross-chain protocol."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    BINANCE = "binance"
    POLYGON = "polygon"
    SOLANA = "solana"
    CUSTOM = "custom"


class NetworkCondition(Enum):
    """Network condition states for adaptive behavior."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class AdaptiveTransactionRouter:
    """
    Implements adaptive routing protocol that dynamically routes transactions
    based on risk assessment and network conditions with Byzantine fault tolerance.
    """
    
    def __init__(
        self,
        detection_model: nn.Module,
        network_topology: Dict[str, List[str]],
        confidence_threshold: float = 0.8,
        verification_threshold: float = 0.6,
        max_path_length: int = 5,
        routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        byzantine_threshold: float = 0.33,
        enable_qos: bool = True
    ):
        self.detection_model = detection_model
        self.network_topology = network_topology
        self.confidence_threshold = confidence_threshold
        self.verification_threshold = verification_threshold
        self.max_path_length = max_path_length
        self.routing_strategy = routing_strategy
        self.byzantine_threshold = byzantine_threshold
        self.enable_qos = enable_qos
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize routing components
        self.routing_table = self._initialize_routing_table()
        self.node_loads = defaultdict(float)
        self.path_cache = LRUCache(capacity=10000)
        self.verification_nodes = self._identify_verification_nodes()
        self.byzantine_nodes = set()
        
        # Performance tracking
        self.routing_history = deque(maxlen=10000)
        self.latency_estimator = LatencyEstimator()
        self.load_balancer = LoadBalancer(self.network_topology)
        self.network_monitor = NetworkMonitor()
        
        # QoS management
        self.qos_manager = QoSManager() if enable_qos else None
        
        # Risk assessment network
        self.risk_assessor = RiskAssessmentNetwork(
            input_dim=detection_model.hidden_dim,
            hidden_dim=128
        )
        
        # Network simulation components
        self.network_simulator = NetworkSimulator(network_topology)
        
    def route_transaction(
        self,
        transaction: Transaction,
        network_state: Dict[str, float],
        real_time_constraints: Optional[Dict[str, float]] = None
    ) -> RoutingDecision:
        """
        Determine optimal routing path for a transaction based on risk and network conditions.
        """
        with self._lock:
            # Update network conditions
            current_condition = self._assess_network_condition(network_state)
            
            # Assess transaction risk using the detection model
            risk_score, confidence = self._assess_transaction_risk(transaction)
            transaction.risk_score = risk_score
            
            # Check for Byzantine nodes and update routing
            self._update_byzantine_detection(network_state)
            
            # Apply QoS if enabled
            if self.enable_qos and real_time_constraints:
                transaction.priority = self.qos_manager.calculate_priority(
                    transaction, real_time_constraints
                )
            
            # Determine routing strategy based on risk level and network condition
            if risk_score > self.confidence_threshold:
                routing_decision = self._route_high_risk_transaction(
                    transaction, network_state, real_time_constraints, current_condition
                )
            elif risk_score > self.verification_threshold:
                routing_decision = self._route_medium_risk_transaction(
                    transaction, network_state, real_time_constraints, current_condition
                )
            else:
                routing_decision = self._route_low_risk_transaction(
                    transaction, network_state, real_time_constraints, current_condition
                )
            
            # Simulate network behavior for evaluation
            simulated_metrics = self.network_simulator.simulate_routing(
                routing_decision, network_state
            )
            
            # Update routing history and node loads
            self._update_routing_state(transaction, routing_decision, simulated_metrics)
            
            return routing_decision
    
    def _assess_network_condition(self, network_state: Dict[str, float]) -> NetworkCondition:
        """Assess overall network condition from state metrics."""
        avg_latency = np.mean([v for k, v in network_state.items() if 'latency' in k])
        avg_loss = np.mean([v for k, v in network_state.items() if 'loss' in k])
        
        if avg_latency < 10 and avg_loss < 0.01:
            return NetworkCondition.EXCELLENT
        elif avg_latency < 50 and avg_loss < 0.05:
            return NetworkCondition.GOOD
        elif avg_latency < 100 and avg_loss < 0.1:
            return NetworkCondition.FAIR
        elif avg_latency < 200 and avg_loss < 0.2:
            return NetworkCondition.POOR
        else:
            return NetworkCondition.CRITICAL
    
    def _assess_transaction_risk(self, transaction: Transaction) -> Tuple[float, float]:
        """Assess transaction risk using the detection model with caching."""
        # Check cache first
        cache_key = self._generate_transaction_hash(transaction)
        cached_result = self.path_cache.get(cache_key)
        
        if cached_result and cached_result['timestamp'] > time.time() - 60:
            return cached_result['risk_score'], cached_result['confidence']
        
        with torch.no_grad():
            # Prepare transaction features
            features = self._prepare_transaction_features(transaction)
            
            # Get risk assessment from model
            risk_output = self.risk_assessor(features)
            risk_score = risk_output['risk_score'].item()
            confidence = risk_output['confidence'].item()
            
            # Cache result
            self.path_cache.put(cache_key, {
                'risk_score': risk_score,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
        return risk_score, confidence
    
    def _prepare_transaction_features(self, transaction: Transaction) -> torch.Tensor:
        """Prepare transaction features for risk assessment."""
        # Basic features
        basic_features = torch.tensor([
            transaction.amount,
            transaction.timestamp,
            float(len(transaction.routing_path)),
            transaction.network_latency,
            float(transaction.retry_count)
        ])
        
        # Combine with transaction attributes
        features = torch.cat([basic_features, transaction.attributes])
        
        return features.unsqueeze(0)  # Add batch dimension
    
    def _update_byzantine_detection(self, network_state: Dict[str, float]):
        """Detect and update Byzantine nodes based on network behavior."""
        for node in self.network_topology:
            # Check for Byzantine behavior indicators
            node_failure_rate = network_state.get(f"{node}_failure_rate", 0.0)
            node_inconsistency = network_state.get(f"{node}_inconsistency", 0.0)
            
            if node_failure_rate > 0.5 or node_inconsistency > 0.3:
                self.byzantine_nodes.add(node)
            elif node in self.byzantine_nodes and node_failure_rate < 0.1:
                self.byzantine_nodes.remove(node)
    
    def _route_high_risk_transaction(
        self,
        transaction: Transaction,
        network_state: Dict[str, float],
        constraints: Optional[Dict[str, float]],
        network_condition: NetworkCondition
    ) -> RoutingDecision:
        """Route high-risk transactions through verification nodes with Byzantine tolerance."""
        source = transaction.source
        destination = transaction.destination
        
        # Find verification path avoiding Byzantine nodes
        verification_path = self._find_byzantine_tolerant_path(
            source, destination, network_state, 
            required_verification_nodes=3
        )
        
        # Find alternative paths for resilience
        alternative_paths = self._find_alternative_paths(
            source, destination, network_state, 
            num_alternatives=2, exclude_path=verification_path
        )
        
        # Estimate routing metrics
        estimated_latency = self.latency_estimator.estimate_path_latency(
            verification_path, network_state
        )
        bandwidth_required = self._calculate_bandwidth_requirement(
            transaction, verification_path, network_condition
        )
        
        # Apply QoS guarantees for high-risk transactions
        qos_guarantee = self._can_guarantee_qos(
            verification_path, constraints, network_state
        ) if constraints else False
        
        routing_decision = RoutingDecision(
            transaction_id=transaction.tx_id,
            path_type="verification",
            path_nodes=verification_path,
            confidence=0.95,
            estimated_latency=estimated_latency,
            bandwidth_required=bandwidth_required,
            priority_level=1,
            alternative_paths=alternative_paths,
            qos_guarantee=qos_guarantee
        )
        
        return routing_decision
    
    def _find_byzantine_tolerant_path(
        self,
        source: str,
        destination: str,
        network_state: Dict[str, float],
        required_verification_nodes: int = 2
    ) -> List[str]:
        """Find path that tolerates Byzantine failures with verification nodes."""
        # Modified A* search with Byzantine awareness
        open_set = [(0, source, [source], 0)]  # (f_score, node, path, verif_count)
        visited = set()
        
        while open_set:
            _, current, path, verif_count = heapq.heappop(open_set)
            
            if current == destination and verif_count >= required_verification_nodes:
                return path
            
            if (current, verif_count) in visited:
                continue
            
            visited.add((current, verif_count))
            
            for neighbor in self.network_topology.get(current, []):
                # Skip Byzantine nodes unless necessary
                if neighbor in self.byzantine_nodes and len(path) > 1:
                    continue
                
                if neighbor not in path and len(path) < self.max_path_length:
                    new_path = path + [neighbor]
                    new_verif_count = verif_count + (1 if neighbor in self.verification_nodes else 0)
                    
                    # Calculate cost with Byzantine penalty
                    edge_cost = self._calculate_edge_cost(
                        current, neighbor, network_state
                    )
                    
                    # Bonus for verification nodes
                    if neighbor in self.verification_nodes:
                        edge_cost *= 0.7
                    
                    g_score = len(new_path) - 1 + edge_cost
                    h_score = self._heuristic_distance(neighbor, destination)
                    f_score = g_score + h_score
                    
                    heapq.heappush(open_set, (f_score, neighbor, new_path, new_verif_count))
        
        # Fallback to shortest path if no Byzantine-tolerant path found
        return self._find_shortest_path(source, destination)
    
    def _find_shortest_path(self, source: str, destination: str) -> List[str]:
        """Find shortest path using Dijkstra's algorithm."""
        if source == destination:
            return [source]
        
        # Use cached routing table if available
        if source in self.routing_table and destination in self.routing_table[source]:
            return self.routing_table[source][destination]
        
        # Dijkstra's algorithm
        distances = {node: float('inf') for node in self.network_topology}
        distances[source] = 0
        previous = {node: None for node in self.network_topology}
        unvisited = set(self.network_topology.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            
            if current == destination:
                break
            
            if distances[current] == float('inf'):
                break
            
            unvisited.remove(current)
            
            for neighbor in self.network_topology.get(current, []):
                if neighbor in unvisited:
                    alt_distance = distances[current] + 1
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        return list(reversed(path)) if path[0] == destination else [source]
    
    def _heuristic_distance(self, node: str, destination: str) -> float:
        """Heuristic distance for A* search (using precomputed distances)."""
        if node == destination:
            return 0.0
        
        # Use shortest path distance as heuristic
        if node in self.routing_table and destination in self.routing_table[node]:
            return float(len(self.routing_table[node][destination]) - 1)
        
        # Default heuristic
        return 1.0
    
    def _calculate_edge_cost(
        self,
        source: str,
        destination: str,
        network_state: Dict[str, float]
    ) -> float:
        """Calculate cost of traversing an edge based on comprehensive network metrics."""
        base_cost = 1.0
        
        edge_key = f"{source}_{destination}"
        
        # Latency component (normalized to 0-1)
        latency = network_state.get(f"{edge_key}_latency", 10.0)
        latency_cost = min(latency / 100.0, 1.0)
        
        # Load component
        dest_load = self.node_loads.get(destination, 0.0)
        load_cost = min(dest_load / 100.0, 1.0)
        
        # Packet loss component
        packet_loss = network_state.get(f"{edge_key}_loss", 0.0)
        loss_cost = packet_loss * 10.0
        
        # Bandwidth utilization component
        bandwidth_util = network_state.get(f"{edge_key}_bandwidth_util", 0.5)
        bandwidth_cost = bandwidth_util
        
        # Byzantine penalty
        byzantine_penalty = 5.0 if destination in self.byzantine_nodes else 0.0
        
        # Weighted combination
        total_cost = (
            base_cost +
            0.3 * latency_cost +
            0.2 * load_cost +
            0.2 * loss_cost +
            0.2 * bandwidth_cost +
            0.1 * byzantine_penalty
        )
        
        return total_cost
    
    def _calculate_bandwidth_requirement(
        self,
        transaction: Transaction,
        path: List[str],
        network_condition: NetworkCondition
    ) -> float:
        """Calculate bandwidth requirement considering network conditions."""
        # Base bandwidth for transaction
        base_bandwidth = 0.1  # Mbps
        
        # Adjust for transaction size
        tx_size_factor = 1.0 + (transaction.attributes.numel() * 4 / 1024 / 1024)  # MB
        
        # Adjust for path length
        path_factor = 1.0 + (len(path) - 1) * 0.1
        
        # Adjust for network condition
        condition_factors = {
            NetworkCondition.EXCELLENT: 1.0,
            NetworkCondition.GOOD: 1.2,
            NetworkCondition.FAIR: 1.5,
            NetworkCondition.POOR: 2.0,
            NetworkCondition.CRITICAL: 3.0
        }
        condition_factor = condition_factors.get(network_condition, 1.5)
        
        # Verification overhead for high-risk transactions
        verification_factor = 1.5 if transaction.risk_score > self.confidence_threshold else 1.0
        
        total_bandwidth = (
            base_bandwidth * 
            tx_size_factor * 
            path_factor * 
            condition_factor * 
            verification_factor
        )
        
        return total_bandwidth
    
    def _can_guarantee_qos(
        self,
        path: List[str],
        constraints: Dict[str, float],
        network_state: Dict[str, float]
    ) -> bool:
        """Check if QoS requirements can be guaranteed for the path."""
        if not self.qos_manager:
            return False
        
        # Check latency constraint
        if 'max_latency' in constraints:
            estimated_latency = self.latency_estimator.estimate_path_latency(
                path, network_state
            )
            if estimated_latency > constraints['max_latency']:
                return False
        
        # Check bandwidth constraint
        if 'min_bandwidth' in constraints:
            min_path_bandwidth = float('inf')
            for i in range(len(path) - 1):
                edge_key = f"{path[i]}_{path[i+1]}"
                available_bandwidth = network_state.get(
                    f"{edge_key}_available_bandwidth", 100.0
                )
                min_path_bandwidth = min(min_path_bandwidth, available_bandwidth)
            
            if min_path_bandwidth < constraints['min_bandwidth']:
                return False
        
        # Check reliability constraint
        if 'max_loss' in constraints:
            path_loss = 1.0
            for i in range(len(path) - 1):
                edge_key = f"{path[i]}_{path[i+1]}"
                edge_reliability = 1.0 - network_state.get(f"{edge_key}_loss", 0.0)
                path_loss *= edge_reliability
            
            total_loss = 1.0 - path_loss
            if total_loss > constraints['max_loss']:
                return False
        
        return True
    
    def _find_alternative_paths(
        self,
        source: str,
        destination: str,
        network_state: Dict[str, float],
        num_alternatives: int = 2,
        exclude_path: List[str] = None
    ) -> List[List[str]]:
        """Find alternative paths for resilience."""
        alternatives = []
        excluded_edges = set()
        
        if exclude_path:
            for i in range(len(exclude_path) - 1):
                excluded_edges.add((exclude_path[i], exclude_path[i+1]))
        
        for _ in range(num_alternatives):
            # Find path avoiding excluded edges
            path = self._find_path_avoiding_edges(
                source, destination, excluded_edges, network_state
            )
            
            if path and path not in alternatives:
                alternatives.append(path)
                # Add this path's edges to excluded set
                for i in range(len(path) - 1):
                    excluded_edges.add((path[i], path[i+1]))
        
        return alternatives
    
    def _find_path_avoiding_edges(
        self,
        source: str,
        destination: str,
        excluded_edges: Set[Tuple[str, str]],
        network_state: Dict[str, float]
    ) -> Optional[List[str]]:
        """Find path avoiding specific edges."""
        # Modified Dijkstra avoiding excluded edges
        distances = {node: float('inf') for node in self.network_topology}
        distances[source] = 0
        previous = {node: None for node in self.network_topology}
        unvisited = set(self.network_topology.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            
            if current == destination:
                break
            
            if distances[current] == float('inf'):
                return None
            
            unvisited.remove(current)
            
            for neighbor in self.network_topology.get(current, []):
                if neighbor in unvisited and (current, neighbor) not in excluded_edges:
                    edge_cost = self._calculate_edge_cost(
                        current, neighbor, network_state
                    )
                    alt_distance = distances[current] + edge_cost
                    
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        if previous.get(destination) is None:
            return None
        
        path = []
        current = destination
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        return list(reversed(path))
    
    def _generate_transaction_hash(self, transaction: Transaction) -> str:
        """Generate hash for transaction caching."""
        hash_input = f"{transaction.tx_id}_{transaction.source}_{transaction.destination}_{transaction.amount}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _update_routing_state(
        self,
        transaction: Transaction,
        routing_decision: RoutingDecision,
        simulated_metrics: Dict[str, float]
    ):
        """Update routing state with thread safety."""
        with self._lock:
            # Update node loads with decay
            for node in routing_decision.path_nodes:
                self.node_loads[node] = self.node_loads[node] * 0.95 + 1.0
            
            # Update latency estimates
            if simulated_metrics:
                for i in range(len(routing_decision.path_nodes) - 1):
                    self.latency_estimator.update_latency(
                        routing_decision.path_nodes[i],
                        routing_decision.path_nodes[i+1],
                        simulated_metrics.get('edge_latencies', [10.0])[i]
                    )
            
            # Update routing history
            self.routing_history.append({
                'timestamp': time.time(),
                'transaction_id': transaction.tx_id,
                'risk_score': transaction.risk_score,
                'routing_decision': routing_decision,
                'network_metrics': simulated_metrics
            })
    
    def _initialize_routing_table(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize routing table using Floyd-Warshall algorithm."""
        nodes = list(self.network_topology.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initialize distance and next hop matrices
        dist = [[float('inf')] * n for _ in range(n)]
        next_hop = [[None] * n for _ in range(n)]
        
        # Set direct edges
        for i in range(n):
            dist[i][i] = 0
            next_hop[i][i] = nodes[i]
        
        for source, neighbors in self.network_topology.items():
            i = node_to_idx[source]
            for neighbor in neighbors:
                j = node_to_idx[neighbor]
                dist[i][j] = 1
                next_hop[i][j] = neighbor
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_hop[i][j] = next_hop[i][k]
        
        # Convert to routing table
        routing_table = {}
        for i, source in enumerate(nodes):
            routing_table[source] = {}
            for j, dest in enumerate(nodes):
                if i != j and next_hop[i][j] is not None:
                    path = self._reconstruct_path(i, j, next_hop, nodes)
                    routing_table[source][dest] = path
        
        return routing_table
    
    def _reconstruct_path(
        self,
        start: int,
        end: int,
        next_hop: List[List[Optional[int]]],
        nodes: List[str]
    ) -> List[str]:
        """Reconstruct path from next hop matrix."""
        if next_hop[start][end] is None:
            return []
        
        path = [nodes[start]]
        current = start
        
        while current != end:
            next_node = next_hop[current][end]
            if next_node is None:
                return []
            current = nodes.index(next_node)
            path.append(nodes[current])
        
        return path
    
    def _identify_verification_nodes(self) -> Set[str]:
        """Identify nodes suitable for transaction verification using multiple criteria."""
        verification_nodes = set()
        
        # Calculate various centrality measures
        degree_centrality = self._calculate_degree_centrality()
        betweenness_centrality = self._calculate_betweenness_centrality()
        
        # Combine centrality measures
        combined_scores = {}
        for node in self.network_topology:
            combined_scores[node] = (
                0.6 * degree_centrality.get(node, 0) +
                0.4 * betweenness_centrality.get(node, 0)
            )
        
        # Select top nodes
        sorted_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        num_verification = max(5, len(sorted_nodes) // 4)
        
        for node, score in sorted_nodes[:num_verification]:
            verification_nodes.add(node)
        
        return verification_nodes
    
    def _calculate_degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for each node."""
        centrality = {}
        total_nodes = len(self.network_topology)
        
        for node, neighbors in self.network_topology.items():
            centrality[node] = len(neighbors) / (total_nodes - 1)
        
        return centrality
    
    def _calculate_betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality using sampling for efficiency."""
        centrality = defaultdict(float)
        nodes = list(self.network_topology.keys())
        
        # Sample subset of node pairs for large graphs
        sample_size = min(100, len(nodes))
        sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
        
        for source in sampled_nodes:
            # Single source shortest paths
            distances, paths = self._single_source_shortest_paths(source)
            
            # Accumulate betweenness
            for target in nodes:
                if target != source and target in paths:
                    path = paths[target]
                    for node in path[1:-1]:  # Exclude source and target
                        centrality[node] += 1.0
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1.0
        for node in centrality:
            centrality[node] /= max_centrality
        
        return dict(centrality)
    
    def _single_source_shortest_paths(self, source: str) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """Compute shortest paths from a single source."""
        distances = {node: float('inf') for node in self.network_topology}
        distances[source] = 0
        paths = {source: [source]}
        
        # Priority queue: (distance, node)
        pq = [(0, source)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current_dist > distances[current]:
                continue
            
            for neighbor in self.network_topology.get(current, []):
                distance = current_dist + 1
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    paths[neighbor] = paths[current] + [neighbor]
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances, paths


class CrossChainCommunicationProtocol:
    """
    Implements cross-chain communication protocol for monitoring transactions
    across multiple blockchain networks with synchronization guarantees.
    """
    
    def __init__(
        self,
        chain_configs: Dict[str, Dict[str, Any]],
        synchronization_interval: float = 1.0,
        consensus_threshold: float = 0.66,
        max_chain_delay: float = 5.0,
        enable_sharding: bool = True
    ):
        self.chain_configs = chain_configs
        self.synchronization_interval = synchronization_interval
        self.consensus_threshold = consensus_threshold
        self.max_chain_delay = max_chain_delay
        self.enable_sharding = enable_sharding
        
        # Thread pool for parallel chain operations
        self.executor = ThreadPoolExecutor(max_workers=len(chain_configs) * 2)
        
        # Initialize chain adapters
        self.chain_adapters = {}
        for chain_id, config in chain_configs.items():
            self.chain_adapters[chain_id] = ChainAdapter(chain_id, config)
        
        # Cross-chain state management
        self.chain_states = {
            chain_id: ChainState(chain_id)
            for chain_id in chain_configs.keys()
        }
        
        # Synchronization components
        self.sync_manager = SynchronizationManager(
            self.chain_states,
            self.consensus_threshold
        )
        self.message_queue = CrossChainMessageQueue()
        self.pending_transactions = defaultdict(list)
        
        # Cross-chain detection components
        self.cross_chain_detector = CrossChainAnomalyDetector()
        self.chain_correlation_analyzer = ChainCorrelationAnalyzer()
        
        # Sharding for scalability
        if enable_sharding:
            self.shard_manager = ShardManager(num_shards=4)
        
        # Performance monitoring
        self.sync_latencies = deque(maxlen=1000)
        self.chain_heights = {chain_id: 0 for chain_id in chain_configs.keys()}
        self.cross_chain_metrics = CrossChainMetrics()
        
        # Bridge and protocol mappings
        self.bridge_registry = BridgeRegistry()
        self.protocol_mappings = self._initialize_protocol_mappings()
    
    async def monitor_cross_chain_transactions(
        self,
        transaction_stream: asyncio.Queue,
        detection_callback: Callable
    ):
        """
        Monitor transactions across multiple chains with real-time synchronization.
        """
        # Start chain monitors
        monitor_tasks = []
        for chain_id in self.chain_adapters:
            task = asyncio.create_task(
                self._monitor_chain(chain_id, transaction_stream)
            )
            monitor_tasks.append(task)
        
        # Start synchronization loop
        sync_task = asyncio.create_task(self._synchronization_loop())
        
        # Start shard processing if enabled
        if self.enable_sharding:
            shard_task = asyncio.create_task(self._shard_processing_loop())
            monitor_tasks.append(shard_task)
        
        # Main detection loop
        try:
            while True:
                # Get transaction from stream with timeout
                try:
                    transaction = await asyncio.wait_for(
                        transaction_stream.get(),
                        timeout=self.synchronization_interval
                    )
                    
                    # Process based on transaction type
                    if self._is_cross_chain_transaction(transaction):
                        # Assign to shard if sharding enabled
                        if self.enable_sharding:
                            shard_id = self.shard_manager.assign_transaction(transaction)
                            await self.shard_manager.process_in_shard(
                                shard_id, transaction, 
                                self._process_cross_chain_transaction
                            )
                        else:
                            detection_result = await self._process_cross_chain_transaction(
                                transaction
                            )
                        
                        # Invoke callback for anomalous transactions
                        if detection_result.get('is_anomalous', False):
                            await detection_callback(transaction, detection_result)
                    
                except asyncio.TimeoutError:
                    # Perform periodic synchronization check
                    await self._check_chain_synchronization()
                    
                except Exception as e:
                    print(f"Error in cross-chain monitoring: {e}")
                    await asyncio.sleep(0.1)
                    
        finally:
            # Cleanup
            for task in monitor_tasks:
                task.cancel()
            sync_task.cancel()
            self.executor.shutdown(wait=True)
    
    async def _monitor_chain(
        self,
        chain_id: str,
        transaction_stream: asyncio.Queue
    ):
        """Monitor individual blockchain for transactions with retry logic."""
        adapter = self.chain_adapters[chain_id]
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                # Fetch latest transactions from chain
                transactions = await adapter.fetch_latest_transactions()
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Update chain state
                self.chain_states[chain_id].update(transactions)
                
                # Process transactions in parallel
                processing_tasks = []
                for tx in transactions:
                    # Add chain metadata
                    tx.chain_id = chain_id
                    
                    # Check for cross-chain indicators
                    if self._has_cross_chain_indicators(tx):
                        processing_tasks.append(
                            asyncio.create_task(self._enrich_transaction(tx))
                        )
                
                # Wait for all enrichment tasks
                if processing_tasks:
                    enriched_txs = await asyncio.gather(*processing_tasks)
                    for tx in enriched_txs:
                        if tx:
                            await transaction_stream.put(tx)
                
                # Update chain height
                if transactions:
                    max_height = max(
                        tx.attributes.get('block_height', 0) 
                        for tx in transactions
                    )
                    self.chain_heights[chain_id] = max(
                        self.chain_heights[chain_id],
                        max_height
                    )
                
                # Update metrics
                self.cross_chain_metrics.update_chain_metrics(
                    chain_id, len(transactions)
                )
                
                # Rate limiting with jitter
                jitter = np.random.uniform(0, 0.1 * adapter.polling_interval)
                await asyncio.sleep(adapter.polling_interval + jitter)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error monitoring chain {chain_id}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"Chain {chain_id} monitor failed repeatedly, marking as unhealthy")
                    self.chain_states[chain_id].mark_unhealthy()
                
                # Exponential backoff with max delay
                backoff_delay = min(2 ** consecutive_errors, 60)
                await asyncio.sleep(backoff_delay)
    
    async def _synchronization_loop(self):
        """Maintain synchronization across chains with consensus mechanism."""
        while True:
            try:
                start_time = time.time()
                
                # Perform synchronization with consensus
                sync_result = await self.sync_manager.synchronize()
                
                # Calculate synchronization latency
                sync_latency = (time.time() - start_time) * 1000
                self.sync_latencies.append(sync_latency)
                
                # Update cross-chain metrics
                self.cross_chain_metrics.update_sync_metrics(sync_result)
                
                # Handle out-of-sync chains
                for chain_id, is_synced in sync_result['chain_status'].items():
                    if not is_synced:
                        await self._handle_out_of_sync_chain(chain_id)
                
                # Check for global consensus
                if sync_result['consensus_achieved']:
                    await self._process_consensus_transactions()
                
                # Adaptive synchronization interval
                if sync_latency > 100:  # If sync is slow
                    adaptive_interval = min(
                        self.synchronization_interval * 1.5,
                        5.0
                    )
                else:
                    adaptive_interval = self.synchronization_interval
                
                await asyncio.sleep(adaptive_interval)
                
            except Exception as e:
                print(f"Synchronization error: {e}")
                await asyncio.sleep(self.synchronization_interval * 2)
    
    async def _process_cross_chain_transaction(
        self,
        transaction: Transaction
    ) -> Dict[str, Any]:
        """Process and analyze cross-chain transaction with comprehensive verification."""
        processing_start = time.time()
        
        # Extract cross-chain components
        chain_components = await self._extract_chain_components(transaction)
        
        # Parallel verification across chains
        verification_tasks = []
        for chain_id in chain_components['involved_chains']:
            verification_tasks.append(
                asyncio.create_task(
                    self._verify_on_chain(transaction, chain_id)
                )
            )
        
        verification_results = await asyncio.gather(*verification_tasks)
        verification_dict = dict(zip(chain_components['involved_chains'], verification_results))
        
        # Analyze for anomalies with ML model
        anomaly_analysis = await self.cross_chain_detector.analyze(
            transaction,
            chain_components,
            verification_dict
        )
        
        # Check correlation patterns across chains
        correlation_analysis = await self.chain_correlation_analyzer.analyze(
            transaction,
            self.chain_states,
            self.cross_chain_metrics.get_recent_patterns()
        )
        
        # Check bridge protocol compliance
        bridge_compliance = await self._verify_bridge_compliance(
            transaction,
            chain_components
        )
        
        # Combine analyses for final decision
        risk_score = self._calculate_cross_chain_risk(
            anomaly_analysis,
            correlation_analysis,
            bridge_compliance,
            verification_dict
        )
        
        processing_time = (time.time() - processing_start) * 1000
        
        detection_result = {
            'is_anomalous': risk_score > 0.7,
            'risk_score': risk_score,
            'anomaly_analysis': anomaly_analysis,
            'correlation_analysis': correlation_analysis,
            'bridge_compliance': bridge_compliance,
            'chain_components': chain_components,
            'verification_results': verification_dict,
            'processing_time_ms': processing_time
        }
        
        # Update metrics
        self.cross_chain_metrics.record_detection(detection_result)
        
        return detection_result
    
    def _is_cross_chain_transaction(self, transaction: Transaction) -> bool:
        """Determine if transaction involves multiple chains with enhanced detection."""
        # Direct cross-chain indicators
        attributes = transaction.attributes
        
        # Check explicit chain identifiers
        if hasattr(attributes, 'source_chain') and hasattr(attributes, 'dest_chain'):
            return attributes.source_chain != attributes.dest_chain
        
        # Check for wrapped token indicators
        if hasattr(attributes, 'token_type'):
            token_type = str(attributes.token_type).lower()
            wrapped_indicators = ['wrapped', 'wbtc', 'weth', 'bridged', 'x', 'multi']
            return any(indicator in token_type for indicator in wrapped_indicators)
        
        # Check destination address patterns
        if self.bridge_registry.is_bridge_address(transaction.destination):
            return True
        
        # Check transaction patterns
        if self._has_bridge_pattern(transaction):
            return True
        
        # Machine learning based detection for subtle patterns
        if hasattr(self, 'ml_detector'):
            ml_prediction = self.ml_detector.predict_cross_chain(transaction)
            return ml_prediction > 0.5
        
        return False
    
    def _has_cross_chain_indicators(self, transaction: Transaction) -> bool:
        """Quick check for potential cross-chain activity."""
        # Amount patterns common in cross-chain
        amount = transaction.amount
        if amount in [1e18, 1e8, 1e6]:  # Common decimal conversions
            return True
        
        # Timing patterns
        if hasattr(transaction, 'timestamp'):
            # Check if timestamp aligns with known bridge intervals
            return self._check_bridge_timing(transaction.timestamp)
        
        return self._is_cross_chain_transaction(transaction)
    
    async def _enrich_transaction(self, transaction: Transaction) -> Optional[Transaction]:
        """Enrich transaction with additional cross-chain metadata."""
        try:
            # Add bridge information if applicable
            if self.bridge_registry.is_bridge_address(transaction.destination):
                bridge_info = self.bridge_registry.get_bridge_info(transaction.destination)
                transaction.attributes = self._add_bridge_metadata(
                    transaction.attributes, bridge_info
                )
            
            # Add cross-chain patterns
            patterns = await self._detect_transaction_patterns(transaction)
            if patterns:
                transaction.attributes = self._add_pattern_metadata(
                    transaction.attributes, patterns
                )
            
            return transaction
            
        except Exception as e:
            print(f"Error enriching transaction {transaction.tx_id}: {e}")
            return None
    
    async def _extract_chain_components(
        self,
        transaction: Transaction
    ) -> Dict[str, Any]:
        """Extract comprehensive components related to different chains."""
        components = {
            'source_chain': transaction.chain_id,
            'dest_chain': None,
            'involved_chains': [transaction.chain_id],
            'bridge_protocol': None,
            'wrapped_tokens': [],
            'intermediate_chains': [],
            'chain_specific_data': {},
            'cross_chain_metadata': {}
        }
        
        # Extract destination chain
        dest_chain = await self._identify_destination_chain(transaction)
        if dest_chain:
            components['dest_chain'] = dest_chain
            components['involved_chains'].append(dest_chain)
        
        # Identify bridge protocol
        components['bridge_protocol'] = await self._identify_bridge_protocol(
            transaction
        )
        
        # Extract wrapped token information
        wrapped_info = await self._extract_wrapped_token_info(transaction)
        components['wrapped_tokens'] = wrapped_info
        
        # Identify intermediate chains for multi-hop transfers
        if components['bridge_protocol']:
            intermediate = await self._identify_intermediate_chains(
                transaction, components['bridge_protocol']
            )
            components['intermediate_chains'] = intermediate
            components['involved_chains'].extend(intermediate)
        
        # Remove duplicates from involved chains
        components['involved_chains'] = list(set(components['involved_chains']))
        
        return components
    
    async def _verify_on_chain(
        self,
        transaction: Transaction,
        chain_id: str
    ) -> Dict[str, Any]:
        """Verify transaction on specific chain with detailed results."""
        adapter = self.chain_adapters.get(chain_id)
        if not adapter:
            return {'verified': False, 'error': 'Chain adapter not found'}
        
        try:
            # Basic verification
            exists = await adapter.verify_transaction(transaction)
            
            if not exists:
                return {'verified': False, 'exists': False}
            
            # Deep verification
            tx_details = await adapter.get_transaction_details(transaction.tx_id)
            
            # Verify amount matches
            amount_matches = abs(tx_details.get('amount', 0) - transaction.amount) < 0.001
            
            # Verify timing
            time_valid = self._verify_transaction_timing(
                transaction.timestamp,
                tx_details.get('timestamp', 0),
                chain_id
            )
            
            # Check for double spending
            double_spend = await adapter.check_double_spend(transaction)
            
            verification_result = {
                'verified': exists and amount_matches and time_valid and not double_spend,
                'exists': exists,
                'amount_matches': amount_matches,
                'time_valid': time_valid,
                'double_spend': double_spend,
                'chain_height': self.chain_heights.get(chain_id, 0),
                'details': tx_details
            }
            
            return verification_result
            
        except Exception as e:
            return {'verified': False, 'error': str(e)}
    
    def _calculate_cross_chain_risk(
        self,
        anomaly_analysis: Dict[str, float],
        correlation_analysis: Dict[str, float],
        bridge_compliance: Dict[str, bool],
        verification_results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate comprehensive cross-chain risk score."""
        # Base scores
        anomaly_score = anomaly_analysis.get('score', 0.5)
        correlation_score = correlation_analysis.get('score', 0.5)
        
        # Bridge compliance factor
        compliance_violations = sum(
            1 for check, passed in bridge_compliance.items() 
            if not passed
        )
        compliance_factor = 1.0 + (compliance_violations * 0.1)
        
        # Verification factor
        verified_chains = sum(
            1 for result in verification_results.values()
            if result.get('verified', False)
        )
        total_chains = len(verification_results)
        verification_factor = 1.0 - (verified_chains / total_chains) if total_chains > 0 else 1.0
        
        # Time synchronization factor
        time_deltas = []
        for chain_id, result in verification_results.items():
            if 'details' in result and 'timestamp' in result['details']:
                time_deltas.append(
                    abs(result['details']['timestamp'] - time.time())
                )
        
        time_sync_factor = 0.0
        if time_deltas:
            max_delta = max(time_deltas)
            if max_delta > self.max_chain_delay:
                time_sync_factor = min((max_delta / self.max_chain_delay) - 1.0, 1.0)
        
        # Weighted combination
        risk_score = (
            0.35 * anomaly_score +
            0.25 * correlation_score +
            0.20 * verification_factor +
            0.10 * (compliance_factor - 1.0) +
            0.10 * time_sync_factor
        )
        
        # Apply compliance factor as multiplier
        risk_score *= compliance_factor
        
        return min(risk_score, 1.0)
    
    async def _verify_bridge_compliance(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Verify transaction compliance with bridge protocols."""
        compliance_checks = {}
        
        bridge_protocol = chain_components.get('bridge_protocol')
        if not bridge_protocol:
            return {'no_bridge': True}
        
        # Get bridge requirements
        bridge_requirements = self.bridge_registry.get_requirements(bridge_protocol)
        
        # Check minimum confirmation requirements
        if 'min_confirmations' in bridge_requirements:
            confirmations = await self._get_transaction_confirmations(transaction)
            compliance_checks['confirmations'] = (
                confirmations >= bridge_requirements['min_confirmations']
            )
        
        # Check amount limits
        if 'min_amount' in bridge_requirements:
            compliance_checks['min_amount'] = (
                transaction.amount >= bridge_requirements['min_amount']
            )
        
        if 'max_amount' in bridge_requirements:
            compliance_checks['max_amount'] = (
                transaction.amount <= bridge_requirements['max_amount']
            )
        
        # Check supported tokens
        if 'supported_tokens' in bridge_requirements:
            token = self._extract_token_from_transaction(transaction)
            compliance_checks['supported_token'] = (
                token in bridge_requirements['supported_tokens']
            )
        
        # Check rate limits
        if 'rate_limit' in bridge_requirements:
            recent_txs = await self._get_recent_bridge_transactions(
                bridge_protocol, 
                transaction.source
            )
            compliance_checks['rate_limit'] = (
                len(recent_txs) < bridge_requirements['rate_limit']
            )
        
        return compliance_checks
    
    async def _handle_out_of_sync_chain(self, chain_id: str):
        """Handle chain that is out of sync with others."""
        print(f"Handling out-of-sync chain: {chain_id}")
        
        # Mark chain as out of sync
        self.chain_states[chain_id].mark_out_of_sync()
        
        # Attempt to resync
        try:
            # Get latest state from other chains
            reference_states = {
                cid: state for cid, state in self.chain_states.items()
                if cid != chain_id and state.is_healthy()
            }
            
            if not reference_states:
                print(f"No healthy reference chains available for {chain_id}")
                return
            
            # Find common ancestor block
            common_height = await self._find_common_ancestor(
                chain_id, reference_states
            )
            
            # Resync from common ancestor
            await self.chain_adapters[chain_id].resync_from_height(common_height)
            
            # Mark as resyncing
            self.chain_states[chain_id].mark_resyncing()
            
        except Exception as e:
            print(f"Failed to resync chain {chain_id}: {e}")
            self.chain_states[chain_id].mark_failed()
    
    async def _process_consensus_transactions(self):
        """Process transactions that have achieved cross-chain consensus."""
        # Get transactions from consensus queue
        consensus_txs = self.sync_manager.get_consensus_transactions()
        
        for tx in consensus_txs:
            # Final verification across all chains
            final_verification = await self._final_cross_chain_verification(tx)
            
            if final_verification['verified']:
                # Mark as processed
                self.sync_manager.mark_processed(tx.tx_id)
                
                # Update cross-chain state
                for chain_id in final_verification['chains']:
                    self.chain_states[chain_id].confirm_transaction(tx)


class StreamingDetectionProtocol:
    """
    Implements streaming detection protocol for real-time transaction analysis
    with sliding windows and incremental learning capabilities.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        update_frequency: int = 100,
        latency_threshold: float = 100.0,  # milliseconds
        memory_efficient: bool = True,
        enable_parallel: bool = True,
        num_workers: int = 4
    ):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.latency_threshold = latency_threshold
        self.memory_efficient = memory_efficient
        self.enable_parallel = enable_parallel
        
        # Sliding window components
        self.transaction_window = SlidingWindow(window_size)
        self.feature_window = FeatureWindow(window_size)
        self.detection_cache = LRUCache(capacity=window_size // 2)
        
        # Streaming statistics
        self.stream_stats = StreamingStatistics()
        self.anomaly_detector = IncrementalAnomalyDetector()
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.detection_latencies = deque(maxlen=1000)
        self.throughput_monitor = ThroughputMonitor()
        
        # Incremental learning components
        self.incremental_model = IncrementalDetectionModel()
        self.concept_drift_detector = ConceptDriftDetector()
        self.model_version = 1
        
        # Parallel processing
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
            self.processing_queue = Queue(maxsize=window_size)
        
        # Real-time processing pipeline
        self.processing_pipeline = self._build_processing_pipeline()
        
        # Adaptive parameters
        self.adaptive_threshold = 0.7
        self.adaptive_window_size = window_size
    
    async def process_transaction_stream(
        self,
        transaction_generator: Callable,
        detection_callback: Callable,
        model: nn.Module
    ):
        """
        Process continuous stream of transactions with real-time detection.
        """
        transaction_count = 0
        last_update = 0
        batch_buffer = []
        batch_size = 32
        
        # Start parallel workers if enabled
        if self.enable_parallel:
            worker_tasks = [
                asyncio.create_task(self._processing_worker(model, detection_callback))
                for _ in range(self.thread_pool._max_workers)
            ]
        
        try:
            async for transaction in transaction_generator():
                start_time = time.time()
                
                try:
                    # Add to sliding window
                    self.transaction_window.add(transaction)
                    
                    # Throughput monitoring
                    self.throughput_monitor.record_transaction()
                    
                    # Batch processing for efficiency
                    batch_buffer.append(transaction)
                    
                    if len(batch_buffer) >= batch_size:
                        # Process batch
                        if self.enable_parallel:
                            await self._process_batch_parallel(
                                batch_buffer, model, detection_callback
                            )
                        else:
                            await self._process_batch(
                                batch_buffer, model, detection_callback
                            )
                        
                        batch_buffer = []
                    
                    # Adaptive window sizing based on stream rate
                    if transaction_count % 100 == 0:
                        self._adapt_window_size()
                    
                    # Check for concept drift
                    if transaction_count % 500 == 0:
                        drift_info = await self._check_and_handle_drift(model)
                        if drift_info['drift_detected']:
                            self.model_version += 1
                    
                    # Incremental model update
                    if transaction_count - last_update >= self.update_frequency:
                        await self._perform_incremental_update(model)
                        last_update = transaction_count
                    
                    # Performance monitoring
                    processing_time = (time.time() - start_time) * 1000
                    self.processing_times.append(processing_time)
                    
                    # Adaptive latency management
                    if processing_time > self.latency_threshold:
                        await self._handle_latency_violation(
                            processing_time, transaction_count
                        )
                    
                    transaction_count += 1
                    
                except Exception as e:
                    print(f"Error processing transaction: {e}")
                    continue
            
            # Process remaining batch
            if batch_buffer:
                await self._process_batch(batch_buffer, model, detection_callback)
                
        finally:
            # Cleanup parallel workers
            if self.enable_parallel:
                for task in worker_tasks:
                    task.cancel()
    
    async def _process_batch(
        self,
        transactions: List[Transaction],
        model: nn.Module,
        detection_callback: Callable
    ):
        """Process a batch of transactions efficiently."""
        # Extract features for batch
        batch_features = []
        for tx in transactions:
            features = self._extract_streaming_features(tx)
            batch_features.append(features)
        
        batch_tensor = torch.stack(batch_features)
        
        # Batch detection
        with torch.no_grad():
            # Get context for batch
            context_features = self.feature_window.get_batch_context(
                len(transactions)
            )
            
            # Prepare streaming input
            streaming_input = self._prepare_batch_streaming_input(
                batch_tensor, context_features
            )
            
            # Run detection
            outputs = model(streaming_input, streaming_mode=True)
            detection_scores = outputs['detection_scores']
        
        # Process results
        for i, (tx, score) in enumerate(zip(transactions, detection_scores)):
            # Cache check
            cache_key = self._generate_cache_key(tx)
            
            # Adaptive threshold
            adaptive_threshold = self._compute_adaptive_threshold()
            
            detection_result = {
                'is_anomalous': score.item() > adaptive_threshold,
                'detection_score': score.item(),
                'adaptive_threshold': adaptive_threshold,
                'model_version': self.model_version,
                'processing_time': time.time()
            }
            
            # Update cache
            self.detection_cache.put(cache_key, detection_result)
            
            # Update statistics
            self.stream_stats.update(tx, detection_result)
            
            # Callback for anomalous transactions
            if detection_result['is_anomalous']:
                await detection_callback(tx, detection_result)
    
    async def _process_batch_parallel(
        self,
        transactions: List[Transaction],
        model: nn.Module,
        detection_callback: Callable
    ):
        """Process batch in parallel using worker threads."""
        # Split batch among workers
        chunk_size = max(1, len(transactions) // self.thread_pool._max_workers)
        chunks = [
            transactions[i:i + chunk_size]
            for i in range(0, len(transactions), chunk_size)
        ]
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._process_batch(chunk, model, detection_callback)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def _processing_worker(
        self,
        model: nn.Module,
        detection_callback: Callable
    ):
        """Worker for parallel transaction processing."""
        while True:
            try:
                # Get transaction from queue
                transaction = await asyncio.get_event_loop().run_in_executor(
                    None, self.processing_queue.get, True, 1.0
                )
                
                # Process transaction
                await self._process_single_transaction(
                    transaction, model, detection_callback
                )
                
            except:
                await asyncio.sleep(0.01)
    
    def _extract_streaming_features(
        self,
        transaction: Transaction
    ) -> torch.Tensor:
        """Extract features optimized for streaming processing."""
        # Get recent window statistics
        window_stats = self.transaction_window.get_statistics()
        
        # Transaction-specific features
        tx_features = torch.tensor([
            transaction.amount,
            transaction.timestamp,
            len(transaction.routing_path),
            transaction.risk_score,
            transaction.network_latency,
            float(transaction.priority),
            float(transaction.retry_count)
        ])
        
        # Normalized amount relative to window
        normalized_amount = (
            transaction.amount / window_stats['mean_amount']
            if window_stats['mean_amount'] > 0 else 1.0
        )
        
        # Window-based features
        window_features = torch.tensor([
            window_stats['mean_amount'],
            window_stats['std_amount'],
            window_stats['transaction_rate'],
            window_stats['anomaly_rate'],
            normalized_amount,
            window_stats['unique_sources'],
            window_stats['unique_destinations']
        ])
        
        # Temporal features
        temporal_features = self._extract_temporal_features(transaction)
        
        # Network features from recent transactions
        network_features = self._extract_network_features(transaction)
        
        # Combine all features
        features = torch.cat([
            tx_features,
            window_features,
            temporal_features,
            network_features
        ])
        
        return features
    
    def _extract_temporal_features(self, transaction: Transaction) -> torch.Tensor:
        """Extract temporal patterns from transaction timing."""
        recent_timestamps = self.transaction_window.get_recent_timestamps(n=10)
        
        if len(recent_timestamps) < 2:
            return torch.zeros(4)
        
        # Time since last transaction
        time_delta = transaction.timestamp - recent_timestamps[-1]
        
        # Average inter-arrival time
        inter_arrival_times = np.diff(recent_timestamps)
        avg_inter_arrival = np.mean(inter_arrival_times) if len(inter_arrival_times) > 0 else 1.0
        
        # Burstiness measure
        if len(inter_arrival_times) > 1:
            burstiness = np.std(inter_arrival_times) / avg_inter_arrival
        else:
            burstiness = 0.0
        
        # Hour of day (cyclical encoding)
        hour = (transaction.timestamp % 86400) / 3600
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        temporal_features = torch.tensor([
            time_delta,
            avg_inter_arrival,
            burstiness,
            hour_sin
        ])
        
        return temporal_features
    
    def _extract_network_features(self, transaction: Transaction) -> torch.Tensor:
        """Extract network-related features from transaction."""
        # Get recent network statistics
        recent_txs = self.transaction_window.get_recent(n=50)
        
        # Source activity
        source_count = sum(1 for tx in recent_txs if tx.source == transaction.source)
        source_amount = sum(tx.amount for tx in recent_txs if tx.source == transaction.source)
        
        # Destination activity
        dest_count = sum(1 for tx in recent_txs if tx.destination == transaction.destination)
        dest_amount = sum(tx.amount for tx in recent_txs if tx.destination == transaction.destination)
        
        # Path similarity with recent transactions
        path_similarity = self._calculate_path_similarity(
            transaction.routing_path, recent_txs
        )
        
        # Network load on path
        path_load = sum(
            self.stream_stats.get_node_load(node)
            for node in transaction.routing_path
        ) / max(len(transaction.routing_path), 1)
        
        network_features = torch.tensor([
            float(source_count),
            source_amount,
            float(dest_count),
            dest_amount,
            path_similarity,
            path_load
        ])
        
        return network_features
    
    def _calculate_path_similarity(
        self,
        path: List[str],
        recent_transactions: List[Transaction]
    ) -> float:
        """Calculate similarity of path with recent transaction paths."""
        if not path or not recent_transactions:
            return 0.0
        
        similarities = []
        for tx in recent_transactions[-10:]:  # Last 10 transactions
            if tx.routing_path:
                # Jaccard similarity
                set1 = set(path)
                set2 = set(tx.routing_path)
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on recent detection history."""
        recent_scores = self.stream_stats.get_recent_scores(n=200)
        
        if len(recent_scores) < 20:
            return self.adaptive_threshold
        
        # Use robust statistics
        scores_array = np.array(recent_scores)
        
        # Remove outliers using IQR
        q1, q3 = np.percentile(scores_array, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_scores = scores_array[
            (scores_array >= lower_bound) & (scores_array <= upper_bound)
        ]
        
        if len(filtered_scores) < 10:
            return self.adaptive_threshold
        
        # Adaptive threshold: high percentile of filtered scores
        adaptive_threshold = np.percentile(filtered_scores, 95)
        
        # Smooth update
        self.adaptive_threshold = (
            0.9 * self.adaptive_threshold + 
            0.1 * adaptive_threshold
        )
        
        # Bound threshold
        return np.clip(self.adaptive_threshold, 0.5, 0.95)
    
    def _adapt_window_size(self):
        """Adapt window size based on stream characteristics."""
        # Get current stream rate
        current_rate = self.throughput_monitor.get_current_rate()
        
        if current_rate > 1000:  # High rate
            # Reduce window size for faster adaptation
            self.adaptive_window_size = max(
                self.window_size // 2,
                500
            )
        elif current_rate < 100:  # Low rate
            # Increase window size for stability
            self.adaptive_window_size = min(
                self.window_size * 2,
                5000
            )
        else:
            # Reset to default
            self.adaptive_window_size = self.window_size
        
        # Update window sizes
        self.transaction_window.resize(self.adaptive_window_size)
        self.feature_window.resize(self.adaptive_window_size)
    
    async def _check_and_handle_drift(self, model: nn.Module) -> Dict[str, Any]:
        """Check for concept drift and handle if detected."""
        # Get recent feature distributions
        recent_features = self.feature_window.get_recent_features(n=200)
        historical_features = self.feature_window.get_historical_features(n=200)
        
        if len(recent_features) < 50 or len(historical_features) < 50:
            return {'drift_detected': False}
        
        # Detect drift
        drift_result = self.concept_drift_detector.detect_drift(
            historical_features,
            recent_features
        )
        
        if drift_result['drift_detected']:
            print(f"Concept drift detected: {drift_result}")
            
            # Handle drift based on severity
            if drift_result['severity'] > 0.8:
                # Severe drift: trigger full retraining
                await self._trigger_full_retraining(model)
            elif drift_result['severity'] > 0.5:
                # Moderate drift: aggressive adaptation
                self.incremental_model.set_aggressive_mode(True)
                await self._perform_aggressive_update(model)
            else:
                # Mild drift: increase learning rate
                self.incremental_model.adapt_learning_rate(factor=2.0)
            
            # Reset statistics
            self.stream_stats.reset_baselines()
            
            # Clear cache
            self.detection_cache.clear()
        
        return drift_result
    
    async def _perform_incremental_update(self, model: nn.Module):
        """Perform incremental model update with recent data."""
        # Get recent labeled data
        recent_data = self.transaction_window.get_recent_labeled(
            n=self.update_frequency
        )
        
        if len(recent_data) < self.update_frequency // 2:
            return
        
        # Prepare mini-batch
        features, labels = self._prepare_incremental_batch(recent_data)
        
        # Perform update
        update_info = await self.incremental_model.update(
            model,
            features,
            labels,
            learning_rate=0.001,
            num_epochs=1
        )
        
        # Log update
        print(f"Incremental update: {update_info}")
        
        # Memory management
        if self.memory_efficient:
            self.transaction_window.clear_old(
                keep_recent=self.adaptive_window_size // 2
            )
            self.feature_window.clear_old(
                keep_recent=self.adaptive_window_size // 2
            )
    
    async def _handle_latency_violation(
        self,
        measured_latency: float,
        transaction_count: int
    ):
        """Handle latency threshold violations."""
        violation_ratio = measured_latency / self.latency_threshold
        
        print(f"Latency violation at tx {transaction_count}: {measured_latency:.2f}ms")
        
        if violation_ratio > 2.0:
            # Severe violation: reduce computational load
            # Temporarily disable expensive features
            self.stream_stats.enable_lightweight_mode()
            
            # Reduce batch size
            self.processing_pipeline = self._build_lightweight_pipeline()
            
        elif violation_ratio > 1.5:
            # Moderate violation: optimize caching
            # Increase cache size
            self.detection_cache.resize(self.detection_cache.capacity * 2)
            
            # Enable more aggressive caching
            self.detection_cache.set_ttl(300)  # 5 minutes
    
    def _build_processing_pipeline(self) -> List[Callable]:
        """Build optimized processing pipeline for streaming."""
        pipeline = [
            self._validate_transaction,
            self._normalize_features,
            self._apply_sliding_window,
            self._detect_anomaly,
            self._update_statistics
        ]
        
        return pipeline
    
    def _build_lightweight_pipeline(self) -> List[Callable]:
        """Build lightweight pipeline for high-load scenarios."""
        pipeline = [
            self._validate_transaction,
            self._detect_anomaly,  # Skip some preprocessing
            self._update_statistics
        ]
        
        return pipeline
    
    def _prepare_incremental_batch(
        self,
        labeled_data: List[Tuple[Transaction, bool]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch for incremental learning."""
        features = []
        labels = []
        
        for transaction, is_anomalous in labeled_data:
            feature_vec = self._extract_streaming_features(transaction)
            features.append(feature_vec)
            labels.append(float(is_anomalous))
        
        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels)
        
        return features_tensor, labels_tensor
    
    def _generate_cache_key(self, transaction: Transaction) -> str:
        """Generate cache key for transaction."""
        # Include model version in cache key
        key_parts = [
            transaction.tx_id,
            str(transaction.amount),
            str(self.model_version),
            str(self.adaptive_threshold)
        ]
        
        return hashlib.sha256('_'.join(key_parts).encode()).hexdigest()[:16]


# Supporting Classes Implementation

class LatencyEstimator:
    """Estimates network latency for routing decisions with learning capability."""
    
    def __init__(self, history_size: int = 1000, learning_rate: float = 0.1):
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.ewma_alpha = learning_rate
        self.current_estimates = {}
        self.confidence_scores = defaultdict(float)
    
    def estimate_path_latency(
        self,
        path: List[str],
        network_state: Dict[str, float]
    ) -> float:
        """Estimate total latency for a path with confidence weighting."""
        if not path or len(path) < 2:
            return 0.0
        
        total_latency = 0.0
        total_confidence = 0.0
        
        for i in range(len(path) - 1):
            edge = f"{path[i]}_{path[i+1]}"
            
            # Use current network state if available
            if f"{edge}_latency" in network_state:
                edge_latency = network_state[f"{edge}_latency"]
                confidence = 1.0  # High confidence for real-time data
            else:
                # Use historical estimate
                edge_latency = self.current_estimates.get(edge, 10.0)
                confidence = self.confidence_scores[edge]
            
            total_latency += edge_latency
            total_confidence += confidence
        
        # Add processing latency estimate (adaptive based on load)
        processing_latency = len(path) * 0.5  # Base 0.5ms per hop
        
        # Adjust for confidence
        if total_confidence > 0:
            confidence_factor = total_confidence / (len(path) - 1)
            total_latency *= (2.0 - confidence_factor)  # Increase estimate for low confidence
        
        return total_latency + processing_latency
    
    def update_latency(
        self,
        source: str,
        destination: str,
        measured_latency: float
    ):
        """Update latency estimates with new measurement."""
        edge = f"{source}_{destination}"
        self.history[edge].append(measured_latency)
        
        # Update EWMA estimate
        if edge in self.current_estimates:
            self.current_estimates[edge] = (
                self.ewma_alpha * measured_latency +
                (1 - self.ewma_alpha) * self.current_estimates[edge]
            )
        else:
            self.current_estimates[edge] = measured_latency
        
        # Update confidence based on variance
        if len(self.history[edge]) >= 5:
            variance = np.var(list(self.history[edge]))
            # Lower variance = higher confidence
            self.confidence_scores[edge] = 1.0 / (1.0 + variance / 100.0)
        else:
            self.confidence_scores[edge] = 0.5


class LoadBalancer:
    """Advanced load balancing for transaction routing."""
    
    def __init__(self, network_topology: Dict[str, List[str]]):
        self.network_topology = network_topology
        self.load_threshold = 0.8
        self.historical_loads = defaultdict(lambda: deque(maxlen=100))
    
    def find_balanced_path(
        self,
        source: str,
        destination: str,
        node_loads: Dict[str, float]
    ) -> List[str]:
        """Find path that optimally balances load across nodes."""
        # Predict future loads
        predicted_loads = self._predict_loads(node_loads)
        
        # Modified Dijkstra with load consideration
        distances = {node: float('infinity') for node in self.network_topology}
        distances[source] = 0
        previous = {node: None for node in self.network_topology}
        unvisited = set(self.network_topology.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            
            if current == destination:
                break
            
            if distances[current] == float('infinity'):
                break
            
            unvisited.remove(current)
            
            for neighbor in self.network_topology.get(current, []):
                if neighbor in unvisited:
                    # Calculate cost with predicted load
                    load_factor = 1.0 + predicted_loads.get(neighbor, 0.0)
                    
                    # Exponential penalty for high loads
                    if predicted_loads.get(neighbor, 0) > self.load_threshold:
                        load_factor *= 2.0
                    
                    cost = 1.0 * load_factor
                    new_distance = distances[current] + cost
                    
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        path = self._reconstruct_path(previous, source, destination)
        
        # Update load predictions
        self._update_load_history(path, node_loads)
        
        return path
    
    def _predict_loads(self, current_loads: Dict[str, float]) -> Dict[str, float]:
        """Predict future loads using historical data."""
        predicted = {}
        
        for node, current_load in current_loads.items():
            history = list(self.historical_loads[node])
            
            if len(history) >= 3:
                # Simple linear prediction
                recent_trend = np.polyfit(range(len(history)), history, 1)[0]
                predicted[node] = current_load + recent_trend * 0.1
            else:
                predicted[node] = current_load
        
        return predicted
    
    def _reconstruct_path(
        self,
        previous: Dict[str, Optional[str]],
        source: str,
        destination: str
    ) -> List[str]:
        """Reconstruct path from previous node mapping."""
        path = []
        current = destination
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        
        return path if path[0] == source else [source]
    
    def _update_load_history(
        self,
        path: List[str],
        current_loads: Dict[str, float]
    ):
        """Update historical load data."""
        for node in path:
            if node in current_loads:
                self.historical_loads[node].append(current_loads[node])


class RiskAssessmentNetwork(nn.Module):
    """Neural network for real-time transaction risk assessment."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(RiskAssessmentNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Multi-layer risk encoder with skip connections
        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.encoder_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Risk and confidence heads
        self.risk_head = nn.Linear(hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
        
        # Auxiliary heads for multi-task learning
        self.amount_predictor = nn.Linear(hidden_dim // 2, 1)
        self.chain_classifier = nn.Linear(hidden_dim // 2, 5)  # 5 chain types
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # First encoding layer
        encoded_1 = self.encoder_1(features)
        
        # Second encoding layer with residual
        encoded_2 = self.encoder_2(encoded_1)
        encoded_2 = encoded_2 + encoded_1  # Skip connection
        
        # Third encoding layer
        encoded_3 = self.encoder_3(encoded_2)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(encoded_3), dim=0)
        attended_features = encoded_3 * attention_weights
        
        # Generate outputs
        risk_score = torch.sigmoid(self.risk_head(attended_features))
        confidence = torch.sigmoid(self.confidence_head(attended_features))
        
        # Auxiliary outputs for regularization
        amount_pred = self.amount_predictor(attended_features)
        chain_logits = self.chain_classifier(attended_features)
        
        return {
            'risk_score': risk_score,
            'confidence': confidence,
            'risk_features': attended_features,
            'attention_weights': attention_weights,
            'amount_prediction': amount_pred,
            'chain_logits': chain_logits
        }


class ChainAdapter:
    """Adapter for interacting with specific blockchain."""
    
    def __init__(self, chain_id: str, config: Dict[str, Any]):
        self.chain_id = chain_id
        self.config = config
        self.polling_interval = config.get('polling_interval', 1.0)
        self.rpc_endpoint = config.get('rpc_endpoint')
        self.connection_pool = []  # Connection pooling
        self.request_cache = LRUCache(capacity=1000)
        self.last_block_height = 0
        
    async def fetch_latest_transactions(self) -> List[Transaction]:
        """Fetch latest transactions from blockchain with caching."""
        try:
            # Check cache first
            cache_key = f"latest_txs_{self.chain_id}_{int(time.time() / self.polling_interval)}"
            cached = self.request_cache.get(cache_key)
            if cached:
                return cached
            
            # Simulate blockchain RPC call
            # In production, this would use actual blockchain APIs
            current_height = self.last_block_height + 1
            
            transactions = []
            for i in range(10):  # Fetch 10 transactions
                tx = Transaction(
                    tx_id=f"{self.chain_id}_{current_height}_{i}",
                    source=f"addr_{np.random.randint(1000)}",
                    destination=f"addr_{np.random.randint(1000)}",
                    amount=np.random.exponential(100),
                    timestamp=time.time(),
                    chain_id=self.chain_id,
                    attributes=torch.randn(10),
                    network_latency=np.random.exponential(10)
                )
                transactions.append(tx)
            
            self.last_block_height = current_height
            
            # Cache results
            self.request_cache.put(cache_key, transactions)
            
            return transactions
            
        except Exception as e:
            print(f"Error fetching transactions from {self.chain_id}: {e}")
            return []
    
    async def verify_transaction(self, transaction: Transaction) -> bool:
        """Verify transaction existence on blockchain."""
        # Simulate verification with 95% success rate
        return np.random.random() > 0.05
    
    async def get_transaction_details(self, tx_id: str) -> Dict[str, Any]:
        """Get detailed transaction information."""
        # Simulate fetching transaction details
        return {
            'tx_id': tx_id,
            'amount': np.random.exponential(100),
            'timestamp': time.time() - np.random.randint(0, 3600),
            'confirmations': np.random.randint(1, 100),
            'block_height': self.last_block_height
        }
    
    async def check_double_spend(self, transaction: Transaction) -> bool:
        """Check if transaction is a double spend."""
        # Simulate double spend detection
        return np.random.random() < 0.01  # 1% double spend rate
    
    async def resync_from_height(self, height: int):
        """Resynchronize from specific block height."""
        print(f"Resyncing {self.chain_id} from height {height}")
        self.last_block_height = max(0, height - 10)  # Go back 10 blocks


class SlidingWindow:
    """Efficient sliding window for streaming transactions."""
    
    def __init__(self, size: int):
        self.size = size
        self.window = deque(maxlen=size)
        self.amount_sum = 0.0
        self.amount_squared_sum = 0.0
        self.unique_sources = set()
        self.unique_destinations = set()
        self.anomaly_count = 0
        
    def add(self, transaction: Transaction):
        """Add transaction to window with incremental statistics update."""
        # Remove oldest if at capacity
        if len(self.window) == self.size:
            old_tx = self.window[0]
            self.amount_sum -= old_tx.amount
            self.amount_squared_sum -= old_tx.amount ** 2
            if old_tx.risk_score > 0.7:
                self.anomaly_count -= 1
        
        # Add new transaction
        self.window.append(transaction)
        self.amount_sum += transaction.amount
        self.amount_squared_sum += transaction.amount ** 2
        
        if transaction.risk_score > 0.7:
            self.anomaly_count += 1
        
        # Update unique sets
        self.unique_sources.add(transaction.source)
        self.unique_destinations.add(transaction.destination)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get window statistics in O(1) time."""
        n = len(self.window)
        if n == 0:
            return {
                'mean_amount': 0.0,
                'std_amount': 0.0,
                'transaction_rate': 0.0,
                'anomaly_rate': 0.0,
                'unique_sources': 0,
                'unique_destinations': 0
            }
        
        mean = self.amount_sum / n
        variance = (self.amount_squared_sum / n) - (mean ** 2)
        std = np.sqrt(max(0, variance))  # Avoid numerical issues
        
        return {
            'mean_amount': mean,
            'std_amount': std,
            'transaction_rate': n / self.size,
            'anomaly_rate': self.anomaly_count / n,
            'unique_sources': len(self.unique_sources),
            'unique_destinations': len(self.unique_destinations)
        }
    
    def get_recent(self, n: int) -> List[Transaction]:
        """Get n most recent transactions."""
        return list(self.window)[-n:]
    
    def get_recent_timestamps(self, n: int) -> List[float]:
        """Get timestamps of recent transactions."""
        return [tx.timestamp for tx in self.get_recent(n)]
    
    def resize(self, new_size: int):
        """Dynamically resize window."""
        if new_size < self.size:
            # Shrink window
            while len(self.window) > new_size:
                self.window.popleft()
        self.size = new_size
        
    def clear_old(self, keep_recent: int):
        """Clear old entries keeping only recent ones."""
        if len(self.window) > keep_recent:
            to_remove = len(self.window) - keep_recent
            for _ in range(to_remove):
                old_tx = self.window.popleft()
                self.amount_sum -= old_tx.amount
                self.amount_squared_sum -= old_tx.amount ** 2
                if old_tx.risk_score > 0.7:
                    self.anomaly_count -= 1


class LRUCache:
    """Least Recently Used cache for detection results."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.ttl_enabled = False
        self.default_ttl = 300  # 5 minutes
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        # Check TTL if enabled
        if self.ttl_enabled:
            value, timestamp = self.cache[key]
            if time.time() - timestamp > self.default_ttl:
                del self.cache[key]
                return None
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return value
        else:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        if key in self.cache:
            # Update existing
            if self.ttl_enabled:
                self.cache[key] = (value, time.time())
            else:
                self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new
            if self.ttl_enabled:
                self.cache[key] = (value, time.time())
            else:
                self.cache[key] = value
            
            # Remove oldest if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
                
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
    
    def resize(self, new_capacity: int):
        """Resize cache capacity."""
        self.capacity = new_capacity
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def set_ttl(self, ttl_seconds: int):
        """Enable TTL with specified duration."""
        self.ttl_enabled = True
        self.default_ttl = ttl_seconds


class ChainState:
    """Maintains state information for a blockchain."""
    
    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.current_height = 0
        self.last_update = time.time()
        self.health_status = "healthy"
        self.sync_status = "synced"
        self.transaction_pool = deque(maxlen=1000)
        self.confirmed_transactions = set()
        self.pending_transactions = set()
        self.metrics = {
            'transactions_per_second': 0.0,
            'average_confirmation_time': 0.0,
            'reorg_count': 0,
            'error_rate': 0.0
        }
        self._lock = threading.Lock()
    
    def update(self, transactions: List[Transaction]):
        """Update chain state with new transactions."""
        with self._lock:
            for tx in transactions:
                self.transaction_pool.append(tx)
                self.pending_transactions.add(tx.tx_id)
                
                # Update height
                tx_height = tx.attributes.get('block_height', 0)
                if tx_height > self.current_height:
                    self.current_height = tx_height
            
            self.last_update = time.time()
            self._update_metrics()
    
    def confirm_transaction(self, transaction: Transaction):
        """Mark transaction as confirmed."""
        with self._lock:
            if transaction.tx_id in self.pending_transactions:
                self.pending_transactions.remove(transaction.tx_id)
                self.confirmed_transactions.add(transaction.tx_id)
    
    def mark_unhealthy(self):
        """Mark chain as unhealthy."""
        self.health_status = "unhealthy"
        self.sync_status = "unknown"
    
    def mark_out_of_sync(self):
        """Mark chain as out of sync."""
        self.sync_status = "out_of_sync"
    
    def mark_resyncing(self):
        """Mark chain as resyncing."""
        self.sync_status = "resyncing"
    
    def mark_failed(self):
        """Mark chain as failed."""
        self.health_status = "failed"
        self.sync_status = "failed"
    
    def is_healthy(self) -> bool:
        """Check if chain is healthy."""
        return self.health_status == "healthy" and self.sync_status == "synced"
    
    def _update_metrics(self):
        """Update chain metrics."""
        if len(self.transaction_pool) > 0:
            # Calculate TPS
            time_window = 60.0  # 1 minute
            recent_txs = [
                tx for tx in self.transaction_pool
                if time.time() - tx.timestamp < time_window
            ]
            self.metrics['transactions_per_second'] = len(recent_txs) / time_window
            
            # Calculate average confirmation time
            confirmation_times = []
            for tx in self.confirmed_transactions:
                # This would track actual confirmation times
                confirmation_times.append(np.random.uniform(1, 10))
            
            if confirmation_times:
                self.metrics['average_confirmation_time'] = np.mean(confirmation_times)


class SynchronizationManager:
    """Manages synchronization across multiple chains."""
    
    def __init__(self, chain_states: Dict[str, ChainState], consensus_threshold: float):
        self.chain_states = chain_states
        self.consensus_threshold = consensus_threshold
        self.consensus_transactions = deque(maxlen=1000)
        self.processed_transactions = set()
        self._lock = threading.Lock()
    
    async def synchronize(self) -> Dict[str, Any]:
        """Perform synchronization across chains."""
        with self._lock:
            # Get current heights
            heights = {
                chain_id: state.current_height
                for chain_id, state in self.chain_states.items()
                if state.is_healthy()
            }
            
            if not heights:
                return {
                    'consensus_achieved': False,
                    'chain_status': {},
                    'latency': 0.0
                }
            
            # Find median height
            median_height = np.median(list(heights.values()))
            
            # Check which chains are in sync
            chain_status = {}
            for chain_id, height in heights.items():
                # Allow some tolerance
                is_synced = abs(height - median_height) <= 5
                chain_status[chain_id] = is_synced
            
            # Check consensus
            synced_chains = sum(1 for synced in chain_status.values() if synced)
            total_chains = len(chain_status)
            consensus_achieved = (synced_chains / total_chains) >= self.consensus_threshold
            
            return {
                'consensus_achieved': consensus_achieved,
                'chain_status': chain_status,
                'median_height': median_height,
                'synced_ratio': synced_chains / total_chains,
                'latency': np.random.uniform(10, 50)  # Simulated latency
            }
    
    def get_consensus_transactions(self) -> List[Transaction]:
        """Get transactions that achieved consensus."""
        with self._lock:
            return list(self.consensus_transactions)
    
    def mark_processed(self, tx_id: str):
        """Mark transaction as processed."""
        with self._lock:
            self.processed_transactions.add(tx_id)


class CrossChainMessageQueue:
    """Message queue for cross-chain communication."""
    
    def __init__(self, max_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.priority_queue = asyncio.PriorityQueue(maxsize=max_size)
        self.message_count = 0
        self.dropped_messages = 0
    
    async def put_message(self, message: Dict[str, Any], priority: int = 5):
        """Put message in queue with priority."""
        try:
            if priority < 5:  # High priority
                await self.priority_queue.put((priority, self.message_count, message))
            else:
                await self.queue.put(message)
            self.message_count += 1
        except asyncio.QueueFull:
            self.dropped_messages += 1
    
    async def get_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get next message from queue."""
        try:
            # Check priority queue first
            if not self.priority_queue.empty():
                _, _, message = await asyncio.wait_for(
                    self.priority_queue.get(), timeout=timeout
                )
                return message
            else:
                return await asyncio.wait_for(
                    self.queue.get(), timeout=timeout
                )
        except asyncio.TimeoutError:
            return None


class CrossChainAnomalyDetector:
    """Detects anomalies in cross-chain transactions."""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=1000)
        self.anomaly_threshold = 0.7
        self.ml_model = self._build_ml_model()
    
    async def analyze(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any],
        verification_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze transaction for cross-chain anomalies."""
        # Extract features
        features = self._extract_anomaly_features(
            transaction, chain_components, verification_results
        )
        
        # ML-based anomaly detection
        with torch.no_grad():
            anomaly_score = self.ml_model(features).item()
        
        # Pattern-based detection
        pattern_score = self._check_known_patterns(transaction, chain_components)
        
        # Timing anomaly detection
        timing_score = self._check_timing_anomalies(transaction, chain_components)
        
        # Amount anomaly detection
        amount_score = self._check_amount_anomalies(transaction, chain_components)
        
        # Combine scores
        combined_score = (
            0.4 * anomaly_score +
            0.3 * pattern_score +
            0.2 * timing_score +
            0.1 * amount_score
        )
        
        return {
            'score': combined_score,
            'ml_score': anomaly_score,
            'pattern_score': pattern_score,
            'timing_score': timing_score,
            'amount_score': amount_score
        }
    
    def _build_ml_model(self) -> nn.Module:
        """Build ML model for anomaly detection."""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _extract_anomaly_features(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any],
        verification_results: Dict[str, Any]
    ) -> torch.Tensor:
        """Extract features for anomaly detection."""
        features = []
        
        # Basic transaction features
        features.extend([
            transaction.amount,
            transaction.network_latency,
            len(transaction.routing_path),
            float(len(chain_components['involved_chains']))
        ])
        
        # Verification features
        verified_count = sum(
            1 for result in verification_results.values()
            if result.get('verified', False)
        )
        features.append(float(verified_count))
        
        # Bridge features
        features.append(1.0 if chain_components['bridge_protocol'] else 0.0)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _check_known_patterns(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any]
    ) -> float:
        """Check for known anomaly patterns."""
        score = 0.0
        
        # Check for suspicious routing patterns
        if len(set(transaction.routing_path)) < len(transaction.routing_path) - 1:
            score += 0.3  # Circular routing
        
        # Check for unusual chain combinations
        chains = chain_components['involved_chains']
        if len(chains) > 3:
            score += 0.2  # Too many chains
        
        return min(score, 1.0)
    
    def _check_timing_anomalies(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any]
    ) -> float:
        """Check for timing anomalies."""
        # Check if transaction is too fast for cross-chain
        if transaction.network_latency < 100 and len(chain_components['involved_chains']) > 1:
            return 0.8  # Suspiciously fast
        
        return 0.0
    
    def _check_amount_anomalies(
        self,
        transaction: Transaction,
        chain_components: Dict[str, Any]
    ) -> float:
        """Check for amount anomalies."""
        # Check for unusual amounts
        if transaction.amount > 1000000:  # Very large
            return 0.5
        elif transaction.amount < 0.0001:  # Dust transaction
            return 0.3
        
        return 0.0


class ChainCorrelationAnalyzer:
    """Analyzes correlations between chains for pattern detection."""
    
    def __init__(self):
        self.correlation_window = deque(maxlen=1000)
        self.correlation_matrix = {}
    
    async def analyze(
        self,
        transaction: Transaction,
        chain_states: Dict[str, ChainState],
        recent_patterns: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze cross-chain correlations."""
        # Update correlation data
        self._update_correlations(transaction, chain_states)
        
        # Calculate correlation score
        correlation_score = self._calculate_correlation_score(
            transaction, chain_states
        )
        
        # Pattern matching score
        pattern_score = self._match_recent_patterns(
            transaction, recent_patterns
        )
        
        # Temporal correlation
        temporal_score = self._calculate_temporal_correlation(
            transaction, chain_states
        )
        
        return {
            'score': (correlation_score + pattern_score + temporal_score) / 3,
            'correlation_score': correlation_score,
            'pattern_score': pattern_score,
            'temporal_score': temporal_score
        }
    
    def _update_correlations(
        self,
        transaction: Transaction,
        chain_states: Dict[str, ChainState]
    ):
        """Update correlation data."""
        self.correlation_window.append({
            'transaction': transaction,
            'chain_heights': {
                chain_id: state.current_height
                for chain_id, state in chain_states.items()
            },
            'timestamp': time.time()
        })
    
    def _calculate_correlation_score(
        self,
        transaction: Transaction,
        chain_states: Dict[str, ChainState]
    ) -> float:
        """Calculate correlation score between chains."""
        # Simple correlation based on transaction patterns
        recent_data = list(self.correlation_window)[-100:]
        
        if len(recent_data) < 10:
            return 0.5
        
        # Calculate transaction frequency correlation
        chain_frequencies = defaultdict(int)
        for data in recent_data:
            chain_frequencies[data['transaction'].chain_id] += 1
        
        # Normalize frequencies
        total = sum(chain_frequencies.values())
        if total == 0:
            return 0.5
        
        for chain in chain_frequencies:
            chain_frequencies[chain] /= total
        
        # Calculate entropy as inverse correlation
        entropy = -sum(
            freq * np.log(freq) if freq > 0 else 0
            for freq in chain_frequencies.values()
        )
        
        # Normalize to 0-1 (lower entropy = higher correlation)
        max_entropy = np.log(len(chain_states))
        correlation_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
        
        return correlation_score
    
    def _match_recent_patterns(
        self,
        transaction: Transaction,
        recent_patterns: List[Dict[str, Any]]
    ) -> float:
        """Match transaction against recent patterns."""
        if not recent_patterns:
            return 0.5
        
        match_scores = []
        for pattern in recent_patterns[-10:]:
            similarity = self._calculate_pattern_similarity(
                transaction, pattern
            )
            match_scores.append(similarity)
        
        return max(match_scores) if match_scores else 0.5
    
    def _calculate_pattern_similarity(
        self,
        transaction: Transaction,
        pattern: Dict[str, Any]
    ) -> float:
        """Calculate similarity between transaction and pattern."""
        similarity = 0.0
        
        # Amount similarity
        if 'amount_range' in pattern:
            min_amt, max_amt = pattern['amount_range']
            if min_amt <= transaction.amount <= max_amt:
                similarity += 0.3
        
        # Chain similarity
        if 'chains' in pattern:
            if transaction.chain_id in pattern['chains']:
                similarity += 0.3
        
        # Timing similarity
        if 'time_pattern' in pattern:
            # Check if transaction matches time pattern
            hour = (transaction.timestamp % 86400) / 3600
            if pattern['time_pattern']['hour_start'] <= hour <= pattern['time_pattern']['hour_end']:
                similarity += 0.4
        
        return similarity
    
    def _calculate_temporal_correlation(
        self,
        transaction: Transaction,
        chain_states: Dict[str, ChainState]
    ) -> float:
        """Calculate temporal correlation between chains."""
        # Check if transaction timing correlates with chain activity
        chain_state = chain_states.get(transaction.chain_id)
        if not chain_state:
            return 0.5
        
        # Compare transaction rate with chain average
        current_tps = chain_state.metrics['transactions_per_second']
        if current_tps > 0:
            # High activity correlation
            if current_tps > 100:
                return 0.8
            else:
                return 0.5
        
        return 0.3


class FeatureWindow:
    """Manages feature window for streaming detection."""
    
    def __init__(self, size: int):
        self.size = size
        self.features = deque(maxlen=size)
        self.feature_stats = {}
        self._update_lock = threading.Lock()
    
    def add(self, features: torch.Tensor):
        """Add features to window."""
        with self._update_lock:
            self.features.append(features)
            self._update_statistics()
    
    def get_context_features(self, k: int) -> torch.Tensor:
        """Get context features from last k entries."""
        with self._update_lock:
            if len(self.features) < k:
                # Pad with zeros if not enough features
                context = list(self.features)
                while len(context) < k:
                    context.append(torch.zeros_like(self.features[0]))
            else:
                context = list(self.features)[-k:]
        
        return torch.stack(context)
    
    def get_batch_context(self, batch_size: int) -> torch.Tensor:
        """Get context for batch processing."""
        with self._update_lock:
            # Use mean of recent features as context
            if len(self.features) >= 10:
                recent = list(self.features)[-10:]
                context = torch.stack(recent).mean(dim=0)
            else:
                context = torch.zeros_like(self.features[0]) if self.features else torch.zeros(20)
        
        # Repeat for batch
        return context.unsqueeze(0).repeat(batch_size, 1)
    
    def get_recent_features(self, n: int) -> np.ndarray:
        """Get recent features as numpy array."""
        with self._update_lock:
            recent = list(self.features)[-n:]
            if recent:
                return torch.stack(recent).numpy()
            return np.array([])
    
    def get_historical_features(self, n: int) -> np.ndarray:
        """Get historical features."""
        with self._update_lock:
            if len(self.features) > n * 2:
                historical = list(self.features)[:-n]
                sample_indices = np.random.choice(
                    len(historical), 
                    size=min(n, len(historical)), 
                    replace=False
                )
                sampled = [historical[i] for i in sample_indices]
                return torch.stack(sampled).numpy()
            return self.get_recent_features(n)
    
    def resize(self, new_size: int):
        """Resize feature window."""
        with self._update_lock:
            self.size = new_size
            # Create new deque with new size
            new_features = deque(list(self.features)[-new_size:], maxlen=new_size)
            self.features = new_features
    
    def clear_old(self, keep_recent: int):
        """Clear old features."""
        with self._update_lock:
            if len(self.features) > keep_recent:
                recent = list(self.features)[-keep_recent:]
                self.features.clear()
                self.features.extend(recent)
    
    def _update_statistics(self):
        """Update feature statistics."""
        if len(self.features) > 0:
            features_array = torch.stack(list(self.features))
            self.feature_stats = {
                'mean': features_array.mean(dim=0),
                'std': features_array.std(dim=0),
                'min': features_array.min(dim=0)[0],
                'max': features_array.max(dim=0)[0]
            }


class StreamingStatistics:
    """Maintains streaming statistics for detection."""
    
    def __init__(self):
        self.total_transactions = 0
        self.anomalous_transactions = 0
        self.recent_scores = deque(maxlen=1000)
        self.node_activity = defaultdict(lambda: {'count': 0, 'last_seen': 0})
        self.lightweight_mode = False
        self.baselines = {
            'mean_score': 0.5,
            'std_score': 0.1,
            'anomaly_rate': 0.01
        }
    
    def update(self, transaction: Transaction, detection_result: Dict[str, Any]):
        """Update statistics with new detection result."""
        self.total_transactions += 1
        
        if detection_result['is_anomalous']:
            self.anomalous_transactions += 1
        
        self.recent_scores.append(detection_result['detection_score'])
        
        # Update node activity
        current_time = time.time()
        self.node_activity[transaction.source]['count'] += 1
        self.node_activity[transaction.source]['last_seen'] = current_time
        self.node_activity[transaction.destination]['count'] += 1
        self.node_activity[transaction.destination]['last_seen'] = current_time
        
        # Update baselines periodically
        if self.total_transactions % 100 == 0:
            self._update_baselines()
    
    def get_recent_scores(self, n: int) -> List[float]:
        """Get n most recent detection scores."""
        return list(self.recent_scores)[-n:]
    
    def get_node_load(self, node: str) -> float:
        """Get current load for a node."""
        if node not in self.node_activity:
            return 0.0
        
        # Calculate load based on recent activity
        node_info = self.node_activity[node]
        time_since_last = time.time() - node_info['last_seen']
        
        # Decay factor
        decay = np.exp(-time_since_last / 300)  # 5 minute decay
        
        # Load based on count and recency
        load = (node_info['count'] / max(self.total_transactions, 1)) * decay
        
        return min(load * 100, 100.0)  # Normalize to 0-100
    
    def enable_lightweight_mode(self):
        """Enable lightweight mode for high-load scenarios."""
        self.lightweight_mode = True
        # Clear some historical data
        self.node_activity.clear()
    
    def reset_baselines(self):
        """Reset statistical baselines."""
        if len(self.recent_scores) > 0:
            self.baselines['mean_score'] = np.mean(self.recent_scores)
            self.baselines['std_score'] = np.std(self.recent_scores)
            self.baselines['anomaly_rate'] = (
                self.anomalous_transactions / max(self.total_transactions, 1)
            )
    
    def _update_baselines(self):
        """Incrementally update baselines."""
        if len(self.recent_scores) >= 100:
            recent_mean = np.mean(list(self.recent_scores)[-100:])
            recent_std = np.std(list(self.recent_scores)[-100:])
            
            # Exponential moving average
            alpha = 0.1
            self.baselines['mean_score'] = (
                alpha * recent_mean + 
                (1 - alpha) * self.baselines['mean_score']
            )
            self.baselines['std_score'] = (
                alpha * recent_std + 
                (1 - alpha) * self.baselines['std_score']
            )


class IncrementalAnomalyDetector:
    """Incremental anomaly detection for streaming data."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.model = self._build_model()
        self.scaler = None
        self.is_fitted = False
    
    def _build_model(self):
        """Build incremental anomaly detection model."""
        # Use Isolation Forest variant for streaming
        return StreamingIsolationForest(
            n_estimators=100,
            contamination=self.contamination
        )
    
    def fit_partial(self, features: np.ndarray):
        """Partially fit the model with new data."""
        if not self.is_fitted:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled)
            self.is_fitted = True
        else:
            features_scaled = self.scaler.transform(features)
            self.model.partial_fit(features_scaled)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            return np.zeros(len(features))
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            return np.zeros(len(features))
        
        features_scaled = self.scaler.transform(features)
        return self.model.score_samples(features_scaled)


class StreamingIsolationForest:
    """Streaming version of Isolation Forest."""
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.trees = []
        self.sample_size = 256
        self.samples_seen = 0
    
    def fit(self, X: np.ndarray):
        """Initial fit of the model."""
        n_samples = len(X)
        
        for _ in range(self.n_estimators):
            # Sample subset
            sample_indices = np.random.choice(
                n_samples, 
                size=min(self.sample_size, n_samples),
                replace=False
            )
            sample = X[sample_indices]
            
            # Build isolation tree
            tree = self._build_tree(sample)
            self.trees.append(tree)
        
        self.samples_seen = n_samples
    
    def partial_fit(self, X: np.ndarray):
        """Incrementally update the model."""
        # Replace oldest trees with new ones
        n_replace = max(1, self.n_estimators // 10)
        
        for i in range(n_replace):
            if len(X) >= self.sample_size:
                sample_indices = np.random.choice(
                    len(X), 
                    size=self.sample_size,
                    replace=False
                )
                sample = X[sample_indices]
            else:
                sample = X
            
            # Build new tree
            new_tree = self._build_tree(sample)
            
            # Replace oldest tree
            if len(self.trees) >= self.n_estimators:
                self.trees.pop(0)
            self.trees.append(new_tree)
        
        self.samples_seen += len(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        scores = self.score_samples(X)
        threshold = np.percentile(scores, self.contamination * 100)
        return np.where(scores < threshold, -1, 1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores."""
        scores = np.zeros(len(X))
        
        for i, x in enumerate(X):
            path_lengths = [self._path_length(x, tree) for tree in self.trees]
            avg_path_length = np.mean(path_lengths)
            
            # Normalize score
            c = self._average_path_length(self.sample_size)
            scores[i] = 2 ** (-avg_path_length / c)
        
        return scores
    
    def _build_tree(self, X: np.ndarray, max_depth: int = 10):
        """Build an isolation tree."""
        return IsolationTreeNode(X, max_depth=max_depth)
    
    def _path_length(self, x: np.ndarray, tree) -> float:
        """Calculate path length for a sample."""
        return tree.path_length(x)
    
    def _average_path_length(self, n: int) -> float:
        """Calculate average path length for n samples."""
        if n <= 1:
            return 1.0
        return 2.0 * (np.log(n - 1) + 0.5772) - 2.0 * (n - 1) / n


class IsolationTreeNode:
    """Node in an isolation tree."""
    
    def __init__(self, X: np.ndarray, depth: int = 0, max_depth: int = 10):
        self.depth = depth
        self.size = len(X)
        self.is_leaf = depth >= max_depth or len(X) <= 1
        
        if not self.is_leaf and len(X) > 1:
            # Select random feature and split value
            self.feature = np.random.randint(X.shape[1])
            min_val = X[:, self.feature].min()
            max_val = X[:, self.feature].max()
            
            if min_val < max_val:
                self.split_value = np.random.uniform(min_val, max_val)
                
                # Split data
                left_mask = X[:, self.feature] < self.split_value
                right_mask = ~left_mask
                
                # Create child nodes
                if np.any(left_mask) and np.any(right_mask):
                    self.left = IsolationTreeNode(X[left_mask], depth + 1, max_depth)
                    self.right = IsolationTreeNode(X[right_mask], depth + 1, max_depth)
                else:
                    self.is_leaf = True
            else:
                self.is_leaf = True
    
    def path_length(self, x: np.ndarray) -> float:
        """Calculate path length for a sample."""
        if self.is_leaf:
            # Adjustment for leaf nodes
            if self.size <= 1:
                return self.depth
            else:
                return self.depth + self._c(self.size)
        
        # Traverse tree
        if x[self.feature] < self.split_value:
            return self.left.path_length(x)
        else:
            return self.right.path_length(x)
    
    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2.0 * (np.log(n - 1) + 0.5772) - 2.0 * (n - 1) / n


class ConceptDriftDetector:
    """Detects concept drift in streaming data."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.test_window = deque(maxlen=window_size)
        self.drift_points = []
    
    def detect_drift(
        self,
        historical_features: np.ndarray,
        recent_features: np.ndarray
    ) -> Dict[str, Any]:
        """Detect concept drift using statistical tests."""
        if len(historical_features) < 10 or len(recent_features) < 10:
            return {'drift_detected': False, 'severity': 0.0}
        
        # Kolmogorov-Smirnov test for each feature
        p_values = []
        for i in range(historical_features.shape[1]):
            hist_feature = historical_features[:, i]
            recent_feature = recent_features[:, i]
            
            # KS test
            ks_stat, p_value = self._ks_test(hist_feature, recent_feature)
            p_values.append(p_value)
        
        # Check if drift detected
        significant_features = sum(1 for p in p_values if p < self.threshold)
        drift_ratio = significant_features / len(p_values)
        
        drift_detected = drift_ratio > 0.3  # 30% of features show drift
        
        # Calculate severity
        if drift_detected:
            severity = min(drift_ratio * 2, 1.0)  # Scale to 0-1
        else:
            severity = drift_ratio
        
        # Additional drift detection methods
        mmd_drift = self._mmd_test(historical_features, recent_features)
        hellinger_drift = self._hellinger_distance(historical_features, recent_features)
        
        # Combine results
        combined_severity = (severity + mmd_drift + hellinger_drift) / 3
        
        result = {
            'drift_detected': drift_detected or combined_severity > 0.5,
            'severity': combined_severity,
            'ks_drift_ratio': drift_ratio,
            'mmd_score': mmd_drift,
            'hellinger_score': hellinger_drift,
            'drifted_features': [i for i, p in enumerate(p_values) if p < self.threshold]
        }
        
        if result['drift_detected']:
            self.drift_points.append(time.time())
        
        return result
    
    def get_drift_severity(self) -> float:
        """Get current drift severity based on recent detections."""
        if not self.drift_points:
            return 0.0
        
        # Count recent drift detections
        current_time = time.time()
        recent_drifts = sum(
            1 for t in self.drift_points 
            if current_time - t < 3600  # Last hour
        )
        
        # Normalize to 0-1
        severity = min(recent_drifts / 10, 1.0)
        
        return severity
    
    def _ks_test(self, sample1: np.ndarray, sample2: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test."""
        # Simple implementation
        n1, n2 = len(sample1), len(sample2)
        
        # Combine and sort
        combined = np.concatenate([sample1, sample2])
        sorted_idx = np.argsort(combined)
        
        # Calculate ECDFs
        ecdf1 = np.zeros(len(combined))
        ecdf2 = np.zeros(len(combined))
        
        for i, idx in enumerate(sorted_idx):
            if idx < n1:
                ecdf1[i:] += 1 / n1
            else:
                ecdf2[i:] += 1 / n2
        
        # KS statistic
        ks_stat = np.max(np.abs(ecdf1 - ecdf2))
        
        # Approximate p-value
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * en * en * ks_stat * ks_stat)
        
        return ks_stat, p_value
    
    def _mmd_test(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Maximum Mean Discrepancy test."""
        # Simple linear MMD
        mean_X = np.mean(X, axis=0)
        mean_Y = np.mean(Y, axis=0)
        
        mmd = np.linalg.norm(mean_X - mean_Y)
        
        # Normalize
        scale = np.mean([np.linalg.norm(mean_X), np.linalg.norm(mean_Y)])
        if scale > 0:
            mmd /= scale
        
        return min(mmd, 1.0)
    
    def _hellinger_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Hellinger distance between distributions."""
        # Discretize continuous features
        n_bins = 10
        distances = []
        
        for i in range(X.shape[1]):
            # Create histograms
            x_min = min(X[:, i].min(), Y[:, i].min())
            x_max = max(X[:, i].max(), Y[:, i].max())
            
            bins = np.linspace(x_min, x_max, n_bins + 1)
            
            hist_X, _ = np.histogram(X[:, i], bins=bins, density=True)
            hist_Y, _ = np.histogram(Y[:, i], bins=bins, density=True)
            
            # Normalize
            hist_X = hist_X / hist_X.sum()
            hist_Y = hist_Y / hist_Y.sum()
            
            # Hellinger distance
            bc = np.sum(np.sqrt(hist_X * hist_Y))
            h_dist = np.sqrt(1 - bc)
            
            distances.append(h_dist)
        
        return np.mean(distances)


class IncrementalDetectionModel:
    """Manages incremental learning for detection model."""
    
    def __init__(self):
        self.update_count = 0
        self.base_lr = 0.001
        self.current_lr = self.base_lr
        self.aggressive_mode = False
        self.optimizer = None
    
    async def update(
        self,
        model: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = None,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """Perform incremental update of the model."""
        if learning_rate is None:
            learning_rate = self.current_lr
        
        # Initialize optimizer if needed
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        model.train()
        total_loss = 0.0
        
        # Create data loader for mini-batch updates
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(32, len(features)),
            shuffle=True
        )
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_labels in loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_features, streaming_mode=True)
                
                # Calculate loss
                loss = F.binary_cross_entropy(
                    outputs['detection_scores'].squeeze(),
                    batch_labels
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
        
        self.update_count += 1
        
        # Decay learning rate
        if not self.aggressive_mode:
            self.current_lr *= 0.99
        
        model.eval()
        
        return {
            'loss': total_loss / (num_epochs * len(loader)),
            'learning_rate': learning_rate,
            'update_count': self.update_count
        }
    
    def set_aggressive_mode(self, aggressive: bool):
        """Set aggressive learning mode for drift handling."""
        self.aggressive_mode = aggressive
        if aggressive:
            self.current_lr = self.base_lr * 2
        else:
            self.current_lr = self.base_lr
    
    def adapt_learning_rate(self, factor: float):
        """Adapt learning rate by factor."""
        self.current_lr *= factor
        self.current_lr = min(self.current_lr, 0.01)  # Cap at 0.01
        self.current_lr = max(self.current_lr, 1e-5)  # Floor at 1e-5


class QoSManager:
    """Manages Quality of Service for transactions."""
    
    def __init__(self):
        self.priority_queues = {
            1: PriorityQueue(),  # Highest priority
            2: PriorityQueue(),
            3: PriorityQueue(),  # Lowest priority
        }
        self.qos_metrics = defaultdict(dict)
    
    def calculate_priority(
        self,
        transaction: Transaction,
        constraints: Dict[str, float]
    ) -> int:
        """Calculate transaction priority based on constraints."""
        priority_score = 0.0
        
        # Latency sensitivity
        if 'max_latency' in constraints:
            if constraints['max_latency'] < 50:  # Very low latency
                priority_score += 3.0
            elif constraints['max_latency'] < 100:
                priority_score += 2.0
            else:
                priority_score += 1.0
        
        # Amount-based priority
        if transaction.amount > 10000:
            priority_score += 2.0
        elif transaction.amount > 1000:
            priority_score += 1.0
        
        # Risk-based priority
        if transaction.risk_score > 0.8:
            priority_score += 3.0  # High-risk needs immediate attention
        
        # Convert to priority level (1-3)
        if priority_score >= 5:
            return 1
        elif priority_score >= 3:
            return 2
        else:
            return 3
    
    def enforce_qos(
        self,
        transaction_id: str,
        constraints: Dict[str, float],
        metrics: Dict[str, float]
    ) -> bool:
        """Check if QoS constraints are being met."""
        violations = []
        
        # Check latency
        if 'max_latency' in constraints:
            if metrics.get('latency', 0) > constraints['max_latency']:
                violations.append('latency')
        
        # Check throughput
        if 'min_throughput' in constraints:
            if metrics.get('throughput', float('inf')) < constraints['min_throughput']:
                violations.append('throughput')
        
        # Check reliability
        if 'min_reliability' in constraints:
            if metrics.get('reliability', 1.0) < constraints['min_reliability']:
                violations.append('reliability')
        
        # Store QoS metrics
        self.qos_metrics[transaction_id] = {
            'constraints': constraints,
            'metrics': metrics,
            'violations': violations,
            'timestamp': time.time()
        }
        
        return len(violations) == 0


class NetworkMonitor:
    """Monitors network conditions in real-time."""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.current_metrics = {}
        self.alert_thresholds = {
            'latency': 200,  # ms
            'packet_loss': 0.05,  # 5%
            'bandwidth_util': 0.9  # 90%
        }
    
    def update_metrics(self, node_pair: str, metrics: NetworkMetrics):
        """Update network metrics for a node pair."""
        self.metrics_history[node_pair].append({
            'metrics': metrics,
            'timestamp': time.time()
        })
        self.current_metrics[node_pair] = metrics
        
        # Check for alerts
        self._check_alerts(node_pair, metrics)
    
    def get_current_state(self) -> Dict[str, float]:
        """Get current network state."""
        state = {}
        
        for node_pair, metrics in self.current_metrics.items():
            state[f"{node_pair}_latency"] = metrics.latency
            state[f"{node_pair}_loss"] = metrics.packet_loss
            state[f"{node_pair}_bandwidth_util"] = metrics.bandwidth / 100.0  # Normalize
            state[f"{node_pair}_jitter"] = metrics.jitter
        
        return state
    
    def predict_congestion(self, node_pair: str, horizon: int = 5) -> float:
        """Predict network congestion in the near future."""
        history = list(self.metrics_history[node_pair])
        
        if len(history) < 10:
            return 0.5  # Not enough data
        
        # Extract congestion indicators
        congestion_values = [
            h['metrics'].congestion_level 
            for h in history[-20:]
        ]
        
        # Simple linear prediction
        if len(congestion_values) >= 3:
            trend = np.polyfit(range(len(congestion_values)), congestion_values, 1)[0]
            predicted = congestion_values[-1] + trend * horizon
            return min(max(predicted, 0.0), 1.0)
        
        return congestion_values[-1]
    
    def _check_alerts(self, node_pair: str, metrics: NetworkMetrics):
        """Check for network condition alerts."""
        alerts = []
        
        if metrics.latency > self.alert_thresholds['latency']:
            alerts.append(f"High latency on {node_pair}: {metrics.latency}ms")
        
        if metrics.packet_loss > self.alert_thresholds['packet_loss']:
            alerts.append(f"High packet loss on {node_pair}: {metrics.packet_loss*100}%")
        
        if metrics.bandwidth > self.alert_thresholds['bandwidth_util'] * 100:
            alerts.append(f"High bandwidth utilization on {node_pair}")
        
        for alert in alerts:
            print(f"NETWORK ALERT: {alert}")


class NetworkSimulator:
    """Simulates network behavior for evaluation."""
    
    def __init__(self, topology: Dict[str, List[str]]):
        self.topology = topology
        self.base_latencies = self._initialize_latencies()
        self.congestion_model = self._build_congestion_model()
    
    def simulate_routing(
        self,
        routing_decision: RoutingDecision,
        network_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Simulate routing and return metrics."""
        path = routing_decision.path_nodes
        
        # Simulate latency
        total_latency = 0.0
        edge_latencies = []
        
        for i in range(len(path) - 1):
            edge = f"{path[i]}_{path[i+1]}"
            
            # Base latency
            base_latency = self.base_latencies.get(edge, 10.0)
            
            # Add congestion
            congestion_factor = 1.0 + network_state.get(f"{edge}_congestion", 0.0)
            
            # Add random jitter
            jitter = np.random.normal(0, 2)
            
            edge_latency = base_latency * congestion_factor + jitter
            edge_latencies.append(max(edge_latency, 0.1))
            total_latency += edge_latencies[-1]
        
        # Simulate packet loss
        packet_loss = self._simulate_packet_loss(path, network_state)
        
        # Simulate bandwidth consumption
        bandwidth_consumed = routing_decision.bandwidth_required
        
        return {
            'total_latency': total_latency,
            'edge_latencies': edge_latencies,
            'packet_loss': packet_loss,
            'bandwidth_consumed': bandwidth_consumed,
            'success_probability': 1.0 - packet_loss
        }
    
    def _initialize_latencies(self) -> Dict[str, float]:
        """Initialize base latencies for edges."""
        latencies = {}
        
        for node, neighbors in self.topology.items():
            for neighbor in neighbors:
                edge = f"{node}_{neighbor}"
                # Random base latency between 5-20ms
                latencies[edge] = np.random.uniform(5, 20)
        
        return latencies
    
    def _build_congestion_model(self):
        """Build congestion simulation model."""
        # Simple congestion model based on load
        return lambda load: min(load / 100.0, 0.9)  # Max 90% congestion
    
    def _simulate_packet_loss(
        self,
        path: List[str],
        network_state: Dict[str, float]
    ) -> float:
        """Simulate packet loss along path."""
        # Calculate path reliability
        path_reliability = 1.0
        
        for i in range(len(path) - 1):
            edge = f"{path[i]}_{path[i+1]}"
            edge_loss = network_state.get(f"{edge}_loss", 0.001)
            edge_reliability = 1.0 - edge_loss
            path_reliability *= edge_reliability
        
        total_loss = 1.0 - path_reliability
        
        return total_loss


class BridgeRegistry:
    """Registry of known bridge protocols and addresses."""
    
    def __init__(self):
        self.bridges = {
            'ethereum_bridge': {
                'addresses': ['0xbridge1', '0xbridge2'],
                'supported_chains': ['ethereum', 'polygon', 'binance'],
                'requirements': {
                    'min_confirmations': 12,
                    'min_amount': 0.01,
                    'max_amount': 1000,
                    'rate_limit': 100  # per hour
                }
            },
            'bitcoin_bridge': {
                'addresses': ['bc1bridge1', 'bc1bridge2'],
                'supported_chains': ['bitcoin', 'ethereum'],
                'requirements': {
                    'min_confirmations': 6,
                    'min_amount': 0.001,
                    'max_amount': 100,
                    'rate_limit': 50
                }
            }
        }
        self.address_to_bridge = self._build_address_map()
    
    def _build_address_map(self) -> Dict[str, str]:
        """Build mapping from address to bridge."""
        address_map = {}
        for bridge_name, bridge_info in self.bridges.items():
            for address in bridge_info['addresses']:
                address_map[address] = bridge_name
        return address_map
    
    def is_bridge_address(self, address: str) -> bool:
        """Check if address is a known bridge."""
        return address in self.address_to_bridge
    
    def get_bridge_info(self, address: str) -> Optional[Dict[str, Any]]:
        """Get bridge information for address."""
        bridge_name = self.address_to_bridge.get(address)
        if bridge_name:
            return self.bridges[bridge_name]
        return None
    
    def get_requirements(self, bridge_protocol: str) -> Dict[str, Any]:
        """Get requirements for bridge protocol."""
        if bridge_protocol in self.bridges:
            return self.bridges[bridge_protocol]['requirements']
        return {}


class CrossChainMetrics:
    """Tracks metrics for cross-chain operations."""
    
    def __init__(self):
        self.chain_metrics = defaultdict(lambda: {
            'transaction_count': 0,
            'success_rate': 1.0,
            'average_latency': 0.0,
            'error_count': 0
        })
        self.sync_metrics = {
            'successful_syncs': 0,
            'failed_syncs': 0,
            'average_sync_time': 0.0
        }
        self.detection_metrics = {
            'total_detections': 0,
            'true_positives': 0,
            'false_positives': 0,
            'average_processing_time': 0.0
        }
        self.pattern_history = deque(maxlen=1000)
    
    def update_chain_metrics(self, chain_id: str, transaction_count: int):
        """Update metrics for a specific chain."""
        self.chain_metrics[chain_id]['transaction_count'] += transaction_count
    
    def update_sync_metrics(self, sync_result: Dict[str, Any]):
        """Update synchronization metrics."""
        if sync_result['consensus_achieved']:
            self.sync_metrics['successful_syncs'] += 1
        else:
            self.sync_metrics['failed_syncs'] += 1
        
        # Update average sync time
        current_avg = self.sync_metrics['average_sync_time']
        total_syncs = (
            self.sync_metrics['successful_syncs'] + 
            self.sync_metrics['failed_syncs']
        )
        
        new_time = sync_result.get('latency', 0.0)
        self.sync_metrics['average_sync_time'] = (
            (current_avg * (total_syncs - 1) + new_time) / total_syncs
        )
    
    def record_detection(self, detection_result: Dict[str, Any]):
        """Record detection metrics."""
        self.detection_metrics['total_detections'] += 1
        
        # Update processing time
        current_avg = self.detection_metrics['average_processing_time']
        total = self.detection_metrics['total_detections']
        new_time = detection_result.get('processing_time_ms', 0.0)
        
        self.detection_metrics['average_processing_time'] = (
            (current_avg * (total - 1) + new_time) / total
        )
        
        # Store pattern for analysis
        self.pattern_history.append({
            'timestamp': time.time(),
            'risk_score': detection_result.get('risk_score', 0.0),
            'chains': detection_result.get('chain_components', {}).get('involved_chains', [])
        })
    
    def get_recent_patterns(self) -> List[Dict[str, Any]]:
        """Get recent cross-chain patterns."""
        return list(self.pattern_history)[-100:]


class ShardManager:
    """Manages sharding for scalable processing."""
    
    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards = [asyncio.Queue() for _ in range(num_shards)]
        self.shard_loads = [0] * num_shards
    
    def assign_transaction(self, transaction: Transaction) -> int:
        """Assign transaction to a shard."""
        # Hash-based sharding
        hash_value = hash(transaction.tx_id)
        shard_id = hash_value % self.num_shards
        
        # Load balancing override
        min_load_shard = self.shard_loads.index(min(self.shard_loads))
        if self.shard_loads[shard_id] > self.shard_loads[min_load_shard] * 1.5:
            shard_id = min_load_shard
        
        self.shard_loads[shard_id] += 1
        
        return shard_id
    
    async def process_in_shard(
        self,
        shard_id: int,
        transaction: Transaction,
        processing_func: Callable
    ) -> Any:
        """Process transaction in assigned shard."""
        # Add to shard queue
        await self.shards[shard_id].put(transaction)
        
        # Process
        result = await processing_func(transaction)
        
        # Update load
        self.shard_loads[shard_id] = max(0, self.shard_loads[shard_id] - 1)
        
        return result


class ThroughputMonitor:
    """Monitors transaction throughput."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # seconds
        self.timestamps = deque()
        self.current_rate = 0.0
        self.peak_rate = 0.0
    
    def record_transaction(self):
        """Record a transaction timestamp."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        # Remove old timestamps
        cutoff_time = current_time - self.window_size
        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.timestamps.popleft()
        
        # Calculate rate
        self.current_rate = len(self.timestamps) / self.window_size
        self.peak_rate = max(self.peak_rate, self.current_rate)
    
    def get_current_rate(self) -> float:
        """Get current throughput rate."""
        return self.current_rate
    
    def get_statistics(self) -> Dict[str, float]:
        """Get throughput statistics."""
        return {
            'current_rate': self.current_rate,
            'peak_rate': self.peak_rate,
            'transactions_in_window': len(self.timestamps)
        }


class StandardScaler:
    """Standard scaler for feature normalization."""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_samples = 0
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        self.n_samples = len(X)
        
        return (X - self.mean) / self.std
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted parameters."""
        if self.mean is None:
            raise ValueError("Scaler not fitted yet")
        
        return (X - self.mean) / self.std
    
    def partial_fit(self, X: np.ndarray):
        """Incrementally update scaler."""
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-8
            self.n_samples = len(X)
        else:
            # Update mean
            new_n = self.n_samples + len(X)
            new_mean = (self.mean * self.n_samples + np.sum(X, axis=0)) / new_n
            
            # Update std (Welford's algorithm)
            delta = X - self.mean
            delta2 = X - new_mean
            m2 = np.sum(delta * delta2, axis=0)
            
            self.std = np.sqrt((self.std ** 2 * self.n_samples + m2) / new_n) + 1e-8
            
            self.mean = new_mean
            self.n_samples = new_n


# Protocol initialization helper
def initialize_protocols(config: Dict[str, Any]) -> Tuple[
    AdaptiveTransactionRouter,
    CrossChainCommunicationProtocol,
    StreamingDetectionProtocol
]:
    """Initialize all protocol components with configuration."""
    
    # Initialize detection model placeholder
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_dim = 128
            
        def forward(self, x, streaming_mode=False):
            return {'detection_scores': torch.rand(x.shape[0], 1)}
    
    detection_model = DummyModel()
    
    # Initialize routing protocol
    router = AdaptiveTransactionRouter(
        detection_model=detection_model,
        network_topology=config['network_topology'],
        confidence_threshold=config.get('confidence_threshold', 0.8),
        verification_threshold=config.get('verification_threshold', 0.6),
        routing_strategy=RoutingStrategy[config.get('routing_strategy', 'HYBRID')],
        enable_qos=config.get('enable_qos', True)
    )
    
    # Initialize cross-chain protocol
    cross_chain = CrossChainCommunicationProtocol(
        chain_configs=config['chain_configs'],
        synchronization_interval=config.get('sync_interval', 1.0),
        consensus_threshold=config.get('consensus_threshold', 0.66),
        enable_sharding=config.get('enable_sharding', True)
    )
    
    # Initialize streaming protocol
    streaming = StreamingDetectionProtocol(
        window_size=config.get('window_size', 1000),
        update_frequency=config.get('update_frequency', 100),
        latency_threshold=config.get('latency_threshold', 100.0),
        memory_efficient=config.get('memory_efficient', True),
        enable_parallel=config.get('enable_parallel', True)
    )
    
    return router, cross_chain, streaming
