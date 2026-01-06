"""
BEACON: Blockchain Edge-based Anomaly detection with Consensus Over Networks
Main Entry Point and Orchestration Module
"""

import os
import sys
import argparse
import logging
import time
import yaml
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import traceback
import signal
import psutil
import GPUtil
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Import BEACON modules
from beacon_core import (
    BEACONModel, NetworkState, ConsensusAggregator, 
    AggregationStrategy, EdgeBEACON
)
from beacon_protocols import (
    AdaptiveTransactionRouter, CrossChainCommunicationProtocol, 
    StreamingDetectionProtocol, NetworkProtocol
)
from beacon_experiments import (
    ExperimentManager, ExperimentConfig,
    ThreadSafePerformanceMonitor
)
from beacon_utils import (
    create_data_loaders, MetricsCalculator, NetworkTopologyGenerator,
    setup_logging, set_random_seeds, get_available_gpus,
    memory_monitor, ProfilerContext
)


class BEACONOrchestrator:
    """
    Main orchestrator for the BEACON framework, managing distributed training,
    federated learning, edge computing, and comprehensive experiments.
    """
    
    def __init__(self, config_path: str, debug: bool = False):
        """Initialize the BEACON orchestrator with configuration."""
        self.config = self._load_and_validate_config(config_path)
        self.debug = debug
        self.logger = self._setup_logging()
        
        # Distributed training state
        self.device = None
        self.rank = None
        self.world_size = None
        self.is_distributed = False
        self.local_rank = None
        
        # Core components
        self.model = None
        self.edge_models = {}
        self.consensus_aggregator = None
        self.performance_monitor = None
        self.experiment_manager = None
        self.tensorboard_writer = None
        
        # Networking components
        self.network_protocol = None
        self.transaction_router = None
        self.cross_chain_protocol = None
        self.streaming_protocol = None
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.best_score = 0.0
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['model', 'training', 'distributed', 'federated_learning',
                           'network_protocols', 'data', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Set defaults for optional parameters
        config.setdefault('experiment', {
            'name': 'beacon_experiment',
            'seed': 42,
            'deterministic': True
        })
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging for distributed training."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(self.config['paths']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger with rank-specific name if distributed
        logger_name = 'BEACON'
        if self.rank is not None:
            logger_name = f'BEACON_rank{self.rank}'
        
        logger = setup_logging(
            name=logger_name,
            log_file=log_dir / f'beacon_{timestamp}.log',
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        
        return logger
    
    def setup_distributed(self, rank: int, world_size: int, local_rank: int):
        """Initialize distributed training environment with error handling."""
        try:
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config['distributed'].get('master_addr', 'localhost')
            os.environ['MASTER_PORT'] = str(self.config['distributed'].get('master_port', 12355))
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config['distributed'].get('backend', 'nccl'),
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            self.rank = rank
            self.world_size = world_size
            self.local_rank = local_rank
            self.is_distributed = True
            
            # Set device
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            
            # Update logger for distributed
            self.logger = self._setup_logging()
            self.logger.info(f"Initialized distributed training on rank {rank}/{world_size}, "
                           f"local_rank {local_rank}")
            
            # Initialize TensorBoard writer for rank 0
            if rank == 0:
                tb_dir = Path(self.config['paths']['tensorboard_dir'])
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(tb_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def initialize_model(self) -> nn.Module:
        """Initialize the BEACON model with proper configuration."""
        model_config = self.config['model']
        
        try:
            # Create base model
            model = BEACONModel(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=model_config['output_dim'],
                num_edge_nodes=model_config['num_edge_nodes'],
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                consensus_threshold=model_config.get('consensus_threshold', 0.7),
                streaming_window=model_config.get('streaming_window', 1000),
                cross_chain_enabled=model_config.get('cross_chain_enabled', True),
                num_chains=model_config.get('num_chains', 2),
                edge_deployment_mode=model_config.get('edge_deployment_mode', False)
            )
            
            # Move to device
            model = model.to(self.device)
            
            # Load checkpoint if specified
            if self.config.get('checkpoint_path'):
                self._load_checkpoint(model, self.config['checkpoint_path'])
            
            # Wrap with DDP if distributed
            if self.is_distributed:
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=True
                )
            
            self.model = model
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            
            self.logger.info(f"Model initialized with {param_count:,} parameters ({param_size_mb:.2f} MB)")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_federated_learning(self):
        """Initialize federated learning components with edge node simulation."""
        fed_config = self.config['federated_learning']
        
        try:
            # Initialize consensus aggregator
            self.consensus_aggregator = ConsensusAggregator(
                num_nodes=fed_config['num_edge_nodes'],
                hidden_dim=self.config['model']['hidden_dim'],
                byzantine_tolerance=fed_config.get('byzantine_tolerance', 0.2),
                aggregation_strategy=AggregationStrategy[fed_config.get('aggregation_strategy', 'BYZANTINE_ROBUST')],
                consensus_rounds=fed_config.get('consensus_rounds', 3),
                verification_threshold=fed_config.get('verification_threshold', 0.8),
                incentive_enabled=fed_config.get('incentive_enabled', True)
            )
            
            # Create edge models for simulation
            if fed_config.get('simulate_edge_nodes', True):
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                
                for i in range(min(fed_config['num_edge_nodes'], 10)):  # Limit simulation
                    edge_model = base_model.deploy_to_edge(node_id=i)
                    self.edge_models[f'edge_{i}'] = edge_model.to(self.device)
            
            self.logger.info(f"Federated learning setup complete with {fed_config['num_edge_nodes']} edge nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to setup federated learning: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_network_protocols(self):
        """Initialize network-aware protocols and monitoring components."""
        network_config = self.config['network_protocols']
        
        try:
            # Initialize network protocol handler
            self.network_protocol = NetworkProtocol(
                node_id=f"coordinator_rank_{self.rank}",
                encryption_key=None  # Would use real encryption in production
            )
            
            # Initialize adaptive transaction router
            self.transaction_router = AdaptiveTransactionRouter(
                hidden_dim=self.config['model']['hidden_dim'],
                num_routes=network_config.get('num_routes', 3),
                network_state=NetworkState(
                    latency=network_config.get('base_latency', 10.0),
                    bandwidth=network_config.get('base_bandwidth', 100.0),
                    packet_loss=0.0,
                    congestion_level=0.0,
                    topology_changes=0
                )
            )
            
            # Setup cross-chain protocol if enabled
            if self.config['model']['cross_chain_enabled']:
                self.cross_chain_protocol = CrossChainCommunicationProtocol(
                    chains=network_config.get('supported_chains', ['ethereum', 'bitcoin']),
                    sync_interval=network_config.get('sync_interval', 60),
                    confirmation_blocks=network_config.get('confirmation_blocks', {'ethereum': 12, 'bitcoin': 6})
                )
            
            # Initialize streaming detection protocol
            self.streaming_protocol = StreamingDetectionProtocol(
                window_size=network_config.get('streaming_window', 1000),
                update_frequency=network_config.get('update_frequency', 10),
                anomaly_threshold=network_config.get('anomaly_threshold', 0.7)
            )
            
            self.logger.info("Network protocols initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup network protocols: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def setup_optimization(self):
        """Setup optimizer, scheduler, and mixed precision training."""
        train_config = self.config['training']
        
        # Optimizer
        optimizer_type = train_config.get('optimizer', 'adamw').lower()
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config.get('weight_decay', 0.01),
                betas=train_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=train_config['learning_rate'],
                weight_decay=train_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        scheduler_type = train_config.get('scheduler', 'cosine').lower()
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs'],
                eta_min=train_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('step_size', 10),
                gamma=train_config.get('gamma', 0.1)
            )
        
        # Mixed precision training
        if train_config.get('mixed_precision', True):
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        self.logger.info(f"Optimization setup: {optimizer_type} optimizer, {scheduler_type} scheduler")
    
    def train_epoch(
        self, 
        epoch: int, 
        data_loader: torch.utils.data.DataLoader,
        metrics_calculator: MetricsCalculator
    ) -> Tuple[float, Dict[str, float]]:
        """Execute one epoch of training with comprehensive monitoring."""
        self.model.train()
        epoch_loss = 0.0
        batch_times = []
        
        # Performance monitoring
        if self.performance_monitor:
            self.performance_monitor.start()
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", disable=(self.rank != 0))
        
        for batch_idx, batch_data in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # Parse batch data based on dataset format
            if len(batch_data) == 2:
                # Simple format: (features, labels)
                node_features, labels = batch_data
                edge_index = None
                edge_attr = None
                sequences = None
            elif len(batch_data) == 4:
                # Graph format: (node_features, edge_index, edge_attr, labels)
                node_features, edge_index, edge_attr, labels = batch_data
                sequences = None
            elif len(batch_data) == 6:
                # Full format with sequences
                node_features, edge_index, edge_attr, in_sequences, out_sequences, labels = batch_data
                sequences = (in_sequences, out_sequences)
            else:
                raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
            
            # Move data to device
            node_features = node_features.to(self.device)
            labels = labels.to(self.device)
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            if edge_attr is not None:
                edge_attr = edge_attr.to(self.device)
            
            # Simulate network conditions
            network_state = self._simulate_network_conditions()
            
            # Forward pass
            loss, predictions = self._training_step(
                node_features, edge_index, edge_attr, sequences, labels, network_state
            )
            
            # Backward pass
            self._backward_step(loss)
            
            # Update metrics
            epoch_loss += loss.item()
            metrics_calculator.update(predictions.detach(), labels)
            
            # Federated learning update
            if self.config['federated_learning']['enabled']:
                if batch_idx % self.config['federated_learning']['update_frequency'] == 0:
                    self._execute_federated_update(batch_idx)
            
            # Streaming protocol update
            if self.streaming_protocol and batch_idx % 10 == 0:
                self.streaming_protocol.update(node_features, predictions)
            
            # Update progress bar
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            if self.rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'batch_time': f'{batch_time:.3f}s'
                })
            
            # Memory monitoring
            if self.debug and batch_idx % 100 == 0:
                self._log_memory_usage()
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(data_loader)
        epoch_metrics = metrics_calculator.compute()
        
        # Add timing metrics
        epoch_metrics['avg_batch_time'] = np.mean(batch_times)
        epoch_metrics['total_epoch_time'] = sum(batch_times)
        
        # Stop performance monitoring
        if self.performance_monitor:
            perf_metrics = self.performance_monitor.stop()
            epoch_metrics.update(perf_metrics)
        
        return avg_loss, epoch_metrics
    
    def _training_step(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
        sequences: Optional[Tuple[torch.Tensor, torch.Tensor]],
        labels: torch.Tensor,
        network_state: NetworkState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute single training step with mixed precision support."""
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
            # Prepare inputs based on available data
            if sequences is not None:
                in_sequences, out_sequences = sequences
                sequence_lengths = {
                    'in': torch.ones(len(node_features), device=self.device),
                    'out': torch.ones(len(node_features), device=self.device)
                }
            else:
                in_sequences = out_sequences = None
                sequence_lengths = None
            
            # Forward pass through model
            outputs = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                in_sequences=in_sequences,
                out_sequences=out_sequences,
                sequence_lengths=sequence_lengths,
                network_state=network_state,
                streaming_mode=self.config['network_protocols'].get('streaming_enabled', False),
                return_edge_outputs=False
            )
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                predictions, auxiliary_outputs = outputs
                routing_decisions = auxiliary_outputs.get('routing_decisions', [])
            else:
                predictions = outputs
                routing_decisions = []
            
            # Calculate loss
            loss = self._calculate_loss(predictions, labels, routing_decisions, network_state)
        
        return loss, predictions
    
    def _backward_step(self, loss: torch.Tensor):
        """Execute backward pass with gradient clipping and mixed precision."""
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip_norm', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip_norm', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_norm']
                )
            
            self.optimizer.step()
    
    def _calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        routing_decisions: List[torch.Tensor],
        network_state: NetworkState
    ) -> torch.Tensor:
        """Calculate comprehensive loss with network-aware components."""
        # Base classification loss
        ce_loss = nn.functional.cross_entropy(predictions, labels)
        
        # Network efficiency penalty
        efficiency_loss = 0.0
        if routing_decisions and self.config['training'].get('efficiency_weight', 0) > 0:
            # Penalize high-priority routing for efficiency
            for decision in routing_decisions:
                if decision is not None and len(decision) > 0:
                    # Assuming index 2 is high-priority route
                    high_priority_ratio = (decision.argmax(dim=1) == 2).float().mean()
                    efficiency_loss += high_priority_ratio
            
            efficiency_loss *= self.config['training']['efficiency_weight']
        
        # Latency-aware penalty
        latency_penalty = 0.0
        if self.config['training'].get('latency_weight', 0) > 0:
            normalized_latency = network_state.latency / 100.0  # Normalize to ~[0, 1]
            latency_penalty = self.config['training']['latency_weight'] * normalized_latency
        
        # Total loss
        total_loss = ce_loss + efficiency_loss + latency_penalty
        
        return total_loss
    
    def _simulate_network_conditions(self) -> NetworkState:
        """Simulate realistic network conditions for training."""
        sim_config = self.config.get('network_simulation', {})
        
        # Base values
        base_latency = sim_config.get('base_latency', 10.0)
        base_bandwidth = sim_config.get('base_bandwidth', 100.0)
        
        # Add random variations
        latency = base_latency * (1 + np.random.normal(0, 0.2))
        bandwidth = base_bandwidth * (1 - np.random.uniform(0, 0.3))
        packet_loss = np.random.beta(1, 50)  # Usually low, occasionally high
        congestion = np.random.beta(2, 5)  # Moderate congestion
        
        # Ensure valid ranges
        latency = max(0.1, latency)
        bandwidth = max(1.0, bandwidth)
        packet_loss = min(0.5, packet_loss)
        
        return NetworkState(
            latency=latency,
            bandwidth=bandwidth,
            packet_loss=packet_loss,
            congestion_level=congestion,
            topology_changes=np.random.poisson(0.1),
            jitter=latency * 0.1,
            routing_distance=np.random.randint(1, 5)
        )
    
    def _execute_federated_update(self, step: int):
        """Execute federated learning update with edge nodes."""
        if not self.edge_models:
            return
        
        try:
            # Simulate edge node updates
            edge_updates = {}
            for edge_id, edge_model in self.edge_models.items():
                # Simulate local training (in practice, this would be actual local training)
                with torch.no_grad():
                    # Simple weight perturbation to simulate local updates
                    for param in edge_model.parameters():
                        param.data += torch.randn_like(param) * 0.01
                
                edge_updates[edge_id] = edge_model.state_dict()
            
            # Aggregate updates
            if self.consensus_aggregator and len(edge_updates) > 0:
                global_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                
                # Perform consensus aggregation
                aggregated_state, trust_scores = self.consensus_aggregator(
                    edge_updates,
                    global_state,
                    network_states={edge_id: self._simulate_network_conditions() for edge_id in edge_updates}
                )
                
                # Update global model (only on rank 0 in distributed setting)
                if self.rank == 0:
                    if hasattr(self.model, 'module'):
                        self.model.module.load_state_dict(aggregated_state)
                    else:
                        self.model.load_state_dict(aggregated_state)
                
                # Log trust scores
                if self.tensorboard_writer and self.rank == 0:
                    for edge_id, score in trust_scores.items():
                        self.tensorboard_writer.add_scalar(f'federated/{edge_id}_trust', score, step)
        
        except Exception as e:
            self.logger.warning(f"Federated update failed at step {step}: {e}")
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        epoch: int = -1
    ) -> Dict[str, float]:
        """Comprehensive evaluation with network performance metrics."""
        self.model.eval()
        metrics_calculator = MetricsCalculator()
        
        network_metrics = {
            'throughput': [],
            'latency': [],
            'consensus_time': [],
            'routing_efficiency': []
        }
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Evaluating", disable=(self.rank != 0)):
                # Parse batch (same as training)
                if len(batch_data) == 2:
                    node_features, labels = batch_data
                    edge_index = edge_attr = None
                else:
                    # Handle other formats...
                    node_features = batch_data[0]
                    labels = batch_data[-1]
                    edge_index = batch_data[1] if len(batch_data) > 2 else None
                    edge_attr = batch_data[2] if len(batch_data) > 3 else None
                
                # Move to device
                node_features = node_features.to(self.device)
                labels = labels.to(self.device)
                if edge_index is not None:
                    edge_index = edge_index.to(self.device)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(self.device)
                
                # Time the inference
                start_time = time.time()
                
                # Forward pass
                outputs = self.model(
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    streaming_mode=False,
                    return_edge_outputs=True
                )
                
                if isinstance(outputs, tuple):
                    predictions, auxiliary_outputs = outputs
                else:
                    predictions = outputs
                    auxiliary_outputs = {}
                
                # Calculate metrics
                inference_time = (time.time() - start_time) * 1000  # ms
                batch_size = len(labels)
                
                metrics_calculator.update(predictions, labels)
                
                # Network performance metrics
                network_metrics['throughput'].append(batch_size / (inference_time / 1000))
                network_metrics['latency'].append(inference_time / batch_size)
                
                # Routing efficiency if available
                if 'routing_decisions' in auxiliary_outputs:
                    routing_eff = self._calculate_routing_efficiency(auxiliary_outputs['routing_decisions'])
                    network_metrics['routing_efficiency'].append(routing_eff)
        
        # Compute final metrics
        eval_metrics = metrics_calculator.compute()
        
        # Add network performance metrics
        for metric, values in network_metrics.items():
            if values:
                eval_metrics[f'network_{metric}_mean'] = np.mean(values)
                eval_metrics[f'network_{metric}_std'] = np.std(values)
        
        # Synchronize metrics across distributed processes
        if self.is_distributed:
            eval_metrics = self._synchronize_metrics(eval_metrics)
        
        return eval_metrics
    
    def _calculate_routing_efficiency(self, routing_decisions: List[torch.Tensor]) -> float:
        """Calculate routing efficiency metric."""
        if not routing_decisions:
            return 1.0
        
        total_efficiency = 0.0
        for decisions in routing_decisions:
            if decisions is not None and len(decisions) > 0:
                # Efficiency: prefer normal routing (index 0) over verification (1) and high-priority (2)
                normal_ratio = (decisions.argmax(dim=1) == 0).float().mean()
                total_efficiency += normal_ratio.item()
        
        return total_efficiency / len(routing_decisions) if routing_decisions else 1.0
    
    def _synchronize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across distributed processes."""
        if not self.is_distributed:
            return metrics
        
        # Convert metrics to tensor for all-reduce
        metric_names = list(metrics.keys())
        metric_values = torch.tensor(
            [metrics[name] for name in metric_names],
            device=self.device
        )
        
        # All-reduce
        dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)
        metric_values /= self.world_size
        
        # Convert back to dictionary
        synchronized_metrics = {
            name: value.item()
            for name, value in zip(metric_names, metric_values)
        }
        
        return synchronized_metrics
    
    def run_training(self):
        """Main training loop with comprehensive experiment tracking."""
        # Setup components
        self.initialize_model()
        self.setup_optimization()
        self.setup_federated_learning()
        self.setup_network_protocols()
        
        # Initialize performance monitor
        self.performance_monitor = ThreadSafePerformanceMonitor()
        
        # Setup data loaders
        train_loader, val_loader, test_loader = self._setup_data_loaders()
        
        # Initialize experiment manager
        if self.rank == 0:
            self.experiment_manager = ExperimentManager(
                config=self.config,
                device=self.device,
                rank=self.rank
            )
        
        # Training loop
        best_score = 0.0
        metrics_calculator = MetricsCalculator()
        
        for epoch in range(self.config['training']['num_epochs']):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch, train_loader, metrics_calculator)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Validation phase
            if epoch % self.config['training'].get('val_interval', 1) == 0:
                val_metrics = self.evaluate(val_loader, epoch)
                
                # Calculate composite score
                val_score = self._calculate_composite_score(val_metrics)
                
                # Save checkpoint if best
                if val_score > best_score and self.rank == 0:
                    best_score = val_score
                    self._save_checkpoint(epoch, val_score, val_metrics)
                
                # Logging
                if self.rank == 0:
                    self._log_epoch_results(epoch, train_loss, train_metrics, val_metrics, val_score)
        
        # Final evaluation on test set
        if self.rank == 0:
            self.logger.info("Running final evaluation on test set")
            test_metrics = self.evaluate(test_loader)
            
            # Run comprehensive experiments if enabled
            if self.config.get('run_full_experiments', False):
                self.logger.info("Running comprehensive experimental evaluation")
                experiment_results = self.experiment_manager.run_full_experimental_suite(
                    self.model,
                    {'train': train_loader, 'val': val_loader, 'test': test_loader}
                )
            
            self._log_final_results(test_metrics)
        
        # Cleanup
        self._cleanup()
    
    def _setup_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """Setup data loaders with proper distributed sampling."""
        data_config = self.config['data']
        
        # Create datasets
        datasets = create_data_loaders(
            train_path=data_config['train_path'],
            val_path=data_config['val_path'],
            test_path=data_config['test_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            distributed=self.is_distributed,
            rank=self.rank,
            world_size=self.world_size,
            streaming=self.config['network_protocols'].get('streaming_enabled', False)
        )
        
        return datasets['train'], datasets['val'], datasets['test']
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score for model selection."""
        weights = self.config['evaluation'].get('metric_weights', {
            'accuracy': 0.3,
            'f1_score': 0.3,
            'network_throughput_mean': 0.2,
            'network_latency_mean': 0.2
        })
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                # Normalize latency (lower is better)
                if 'latency' in metric:
                    value = 1.0 / (1.0 + value / 100.0)
                score += weight * value
        
        return score
    
    def _save_checkpoint(self, epoch: int, score: float, metrics: Dict[str, float]):
        """Save model checkpoint with comprehensive metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'score': score,
            'metrics': metrics,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
            'git_hash': self._get_git_hash()
        }
        
        checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'beacon_epoch_{epoch:03d}_score_{score:.4f}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best model
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path} "
                        f"(epoch {checkpoint['epoch']}, score {checkpoint['score']:.4f})")
    
    def _log_epoch_results(
        self, 
        epoch: int, 
        train_loss: float, 
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        val_score: float
    ):
        """Log epoch results to console and TensorBoard."""
        # Console logging
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Score={val_score:.4f}, "
            f"Val Acc={val_metrics.get('accuracy', 0):.4f}, "
            f"Val F1={val_metrics.get('f1_score', 0):.4f}"
        )
        
        # TensorBoard logging
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('train/loss', train_loss, epoch)
            self.tensorboard_writer.add_scalar('val/score', val_score, epoch)
            
            for metric, value in train_metrics.items():
                self.tensorboard_writer.add_scalar(f'train/{metric}', value, epoch)
            
            for metric, value in val_metrics.items():
                self.tensorboard_writer.add_scalar(f'val/{metric}', value, epoch)
            
            # Log learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.tensorboard_writer.add_scalar('train/learning_rate', lr, epoch)
    
    def _log_final_results(self, test_metrics: Dict[str, float]):
        """Log final test results."""
        self.logger.info("="*80)
        self.logger.info("FINAL TEST RESULTS")
        self.logger.info("="*80)
        
        for metric, value in sorted(test_metrics.items()):
            self.logger.info(f"{metric}: {value:.4f}")
        
        # Save final results
        results_dir = Path(self.config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / f'final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    def _log_memory_usage(self):
        """Log current memory usage for debugging."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.logger.debug(f"GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        self.logger.debug(f"CPU Memory: {cpu_memory:.2f}GB")
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash for reproducibility."""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return "unknown"
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.is_distributed:
            dist.destroy_process_group()
        
        self.logger.info("Training completed successfully")


class ResourceMonitor:
    """Monitor system resources during training."""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
    
    def update(self):
        """Update resource usage statistics."""
        # GPU memory
        if torch.cuda.is_available():
            current_gpu = torch.cuda.max_memory_allocated() / 1024**3
            self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu)
        
        # CPU memory
        process = psutil.Process()
        current_cpu = process.memory_info().rss / 1024**3
        self.peak_cpu_memory = max(self.peak_cpu_memory, current_cpu)
    
    def get_summary(self) -> Dict[str, float]:
        """Get resource usage summary."""
        return {
            'runtime_hours': (time.time() - self.start_time) / 3600,
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'peak_cpu_memory_gb': self.peak_cpu_memory
        }


def setup_distributed_training(rank: int, world_size: int, config_path: str, args: argparse.Namespace):
    """Setup function for distributed training processes."""
    try:
        # Calculate local rank (for multi-node training)
        local_rank = rank % torch.cuda.device_count()
        
        # Create orchestrator
        orchestrator = BEACONOrchestrator(config_path, debug=args.debug)
        
        # Setup distributed environment
        orchestrator.setup_distributed(rank, world_size, local_rank)
        
        # Set random seeds
        set_random_seeds(
            orchestrator.config['experiment']['seed'] + rank,
            deterministic=orchestrator.config['experiment'].get('deterministic', True)
        )
        
        # Run training
        orchestrator.run_training()
        
    except Exception as e:
        logging.error(f"Error in rank {rank}: {e}")
        logging.error(traceback.format_exc())
        raise


def main():
    """Main entry point for BEACON framework."""
    parser = argparse.ArgumentParser(
        description='BEACON: Blockchain Edge-based Anomaly detection with Consensus Over Networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'experiments', 'profile'],
                       help='Execution mode')
    
    # Distributed training arguments
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--nodes', type=int, default=1,
                       help='Number of nodes for distributed training')
    parser.add_argument('--node-rank', type=int, default=0,
                       help='Rank of the current node')
    parser.add_argument('--master-addr', type=str, default='localhost',
                       help='Master node address')
    parser.add_argument('--master-port', type=int, default=12355,
                       help='Master node port')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for resuming/evaluation')
    
    # Experiment arguments
    parser.add_argument('--experiment-type', type=str, default='full',
                       choices=['full', 'distributed', 'network', 'scalability', 'cross_chain'],
                       help='Type of experiment to run')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--memory-debug', action='store_true',
                       help='Enable memory debugging')
    
    args = parser.parse_args()
    
    # Determine number of GPUs
    if args.gpus is None:
        args.gpus = torch.cuda.device_count()
    
    # Validate GPU availability
    if args.gpus > 0 and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run with --gpus 0 for CPU mode.")
    
    if args.gpus > torch.cuda.device_count():
        logging.warning(f"Requested {args.gpus} GPUs but only {torch.cuda.device_count()} available")
        args.gpus = torch.cuda.device_count()
    
    # Update config with command line arguments
    config = yaml.safe_load(open(args.config))
    if args.master_addr != 'localhost':
        config['distributed']['master_addr'] = args.master_addr
    if args.master_port != 12355:
        config['distributed']['master_port'] = args.master_port
    
    # Save updated config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name
    
    try:
        if args.mode == 'train':
            if args.gpus > 1 or args.nodes > 1:
                # Multi-GPU or multi-node training
                world_size = args.gpus * args.nodes
                
                if args.nodes > 1:
                    # Multi-node: each node spawns processes for its GPUs
                    mp.spawn(
                        setup_distributed_training,
                        args=(world_size, config_path, args),
                        nprocs=args.gpus,
                        join=True
                    )
                else:
                    # Single node, multi-GPU
                    mp.spawn(
                        setup_distributed_training,
                        args=(args.gpus, config_path, args),
                        nprocs=args.gpus,
                        join=True
                    )
            else:
                # Single GPU or CPU training
                orchestrator = BEACONOrchestrator(config_path, debug=args.debug)
                if args.gpus == 1:
                    orchestrator.device = torch.device('cuda:0')
                else:
                    orchestrator.device = torch.device('cpu')
                orchestrator.rank = 0
                orchestrator.world_size = 1
                orchestrator.is_distributed = False
                
                set_random_seeds(
                    config['experiment']['seed'],
                    deterministic=config['experiment'].get('deterministic', True)
                )
                
                orchestrator.run_training()
        
        elif args.mode == 'evaluate':
            # Evaluation mode
            orchestrator = BEACONOrchestrator(config_path, debug=args.debug)
            orchestrator.device = torch.device('cuda:0' if args.gpus > 0 else 'cpu')
            orchestrator.rank = 0
            orchestrator.world_size = 1
            orchestrator.is_distributed = False
            
            # Initialize model and load checkpoint
            orchestrator.initialize_model()
            if args.checkpoint:
                orchestrator._load_checkpoint(orchestrator.model, args.checkpoint)
            else:
                raise ValueError("Checkpoint path required for evaluation mode")
            
            # Run evaluation
            _, _, test_loader = orchestrator._setup_data_loaders()
            test_metrics = orchestrator.evaluate(test_loader)
            
            print("\nTest Results:")
            print("="*50)
            for metric, value in sorted(test_metrics.items()):
                print(f"{metric:30s}: {value:.4f}")
        
        elif args.mode == 'experiments':
            # Run comprehensive experiments
            orchestrator = BEACONOrchestrator(config_path, debug=args.debug)
            orchestrator.device = torch.device('cuda:0' if args.gpus > 0 else 'cpu')
            orchestrator.rank = 0
            orchestrator.world_size = 1
            orchestrator.is_distributed = False
            
            # Initialize components
            orchestrator.initialize_model()
            if args.checkpoint:
                orchestrator._load_checkpoint(orchestrator.model, args.checkpoint)
            
            # Setup experiment manager
            experiment_manager = ExperimentManager(
                config=config,
                device=orchestrator.device,
                rank=0
            )
            
            # Run experiments
            train_loader, val_loader, test_loader = orchestrator._setup_data_loaders()
            datasets = {'train': train_loader, 'val': val_loader, 'test': test_loader}
            
            experiment_results = experiment_manager.run_full_experimental_suite(
                orchestrator.model,
                datasets
            )
            
            print("\nExperiment completed. Results saved to:", experiment_manager.output_dir)
        
        elif args.mode == 'profile':
            # Profiling mode
            with ProfilerContext() as profiler:
                # Run single training epoch with profiling
                orchestrator = BEACONOrchestrator(config_path, debug=True)
                orchestrator.device = torch.device('cuda:0' if args.gpus > 0 else 'cpu')
                orchestrator.rank = 0
                orchestrator.world_size = 1
                orchestrator.is_distributed = False
                
                orchestrator.initialize_model()
                orchestrator.setup_optimization()
                
                train_loader, _, _ = orchestrator._setup_data_loaders()
                metrics_calculator = MetricsCalculator()
                
                # Profile one epoch
                orchestrator.train_epoch(0, train_loader, metrics_calculator)
            
            print(f"\nProfiling results saved to: {profiler.output_path}")
    
    finally:
        # Cleanup temporary config file
        if 'config_path' in locals() and os.path.exists(config_path) and config_path != args.config:
            os.unlink(config_path)


if __name__ == '__main__':
    # Handle keyboard interrupts gracefully
    def signal_handler(sig, frame):
        logging.info("Received interrupt signal. Cleaning up...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run main
    main()
