"""
BEACON Utilities Module
Efficient data loading, network simulation, performance metrics, and streaming preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, to_undirected, add_self_loops, to_dense_batch
from torch_geometric.loader import NeighborLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator, Callable, Set
from pathlib import Path
import networkx as nx
from dataclasses import dataclass, field
from collections import deque, defaultdict, OrderedDict
import mmap
import pickle
import json
import h5py
import time
import random
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
import psutil
from contextlib import contextmanager
import gc
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.sparse as sp
from scipy.stats import ks_2samp, entropy

# Additional imports for distributed processing
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


@dataclass
class TransactionBatch:
    """Optimized batch container for transaction data with comprehensive metadata."""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    in_sequences: torch.Tensor
    out_sequences: torch.Tensor
    sequence_lengths: Dict[str, torch.Tensor]
    labels: torch.Tensor
    timestamps: torch.Tensor
    chain_ids: List[str]
    batch_ptr: Optional[torch.Tensor] = None  # For batched graphs
    node_batch: Optional[torch.Tensor] = None  # Node to graph assignment
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: torch.device) -> 'TransactionBatch':
        """Move batch to specified device."""
        return TransactionBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            in_sequences=self.in_sequences.to(device),
            out_sequences=self.out_sequences.to(device),
            sequence_lengths={k: v.to(device) for k, v in self.sequence_lengths.items()},
            labels=self.labels.to(device),
            timestamps=self.timestamps.to(device),
            chain_ids=self.chain_ids,
            batch_ptr=self.batch_ptr.to(device) if self.batch_ptr is not None else None,
            node_batch=self.node_batch.to(device) if self.node_batch is not None else None,
            metadata=self.metadata
        )
    
    def pin_memory(self) -> 'TransactionBatch':
        """Pin batch memory for faster GPU transfer."""
        return TransactionBatch(
            node_features=self.node_features.pin_memory(),
            edge_index=self.edge_index.pin_memory(),
            edge_attr=self.edge_attr.pin_memory(),
            in_sequences=self.in_sequences.pin_memory(),
            out_sequences=self.out_sequences.pin_memory(),
            sequence_lengths={k: v.pin_memory() for k, v in self.sequence_lengths.items()},
            labels=self.labels.pin_memory(),
            timestamps=self.timestamps.pin_memory(),
            chain_ids=self.chain_ids,
            batch_ptr=self.batch_ptr.pin_memory() if self.batch_ptr is not None else None,
            node_batch=self.node_batch.pin_memory() if self.node_batch is not None else None,
            metadata=self.metadata
        )


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader optimized for 384GB RAM systems.
    Implements advanced caching, memory mapping, and parallel loading strategies.
    """
    
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 128,
        num_workers: int = 32,
        cache_size_gb: float = 100.0,
        prefetch_batches: int = 10,
        use_memory_mapping: bool = True,
        streaming_mode: bool = False,
        pin_memory: bool = True,
        enable_gpu_caching: bool = True
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size_gb = cache_size_gb
        self.prefetch_batches = prefetch_batches
        self.use_memory_mapping = use_memory_mapping
        self.streaming_mode = streaming_mode
        self.pin_memory = pin_memory
        self.enable_gpu_caching = enable_gpu_caching
        
        # Initialize components
        self.cache_manager = CacheManager(max_size_gb=cache_size_gb)
        self.memory_pool = MemoryPool(total_memory_gb=384)
        self.prefetcher = DataPrefetcher(num_workers=min(num_workers, 16))
        
        # GPU cache for frequently accessed data
        if enable_gpu_caching and torch.cuda.is_available():
            self.gpu_cache = GPUCache(max_size_gb=min(10.0, torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.1))
        else:
            self.gpu_cache = None
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.num_samples = self.metadata['num_samples']
        self.feature_dim = self.metadata['feature_dim']
        
        # Initialize data access
        if use_memory_mapping:
            self.data_accessor = MemoryMappedAccessor(data_path)
        else:
            self.data_accessor = DirectAccessor(data_path)
        
        # Streaming components
        if streaming_mode:
            self.stream_processor = StreamProcessor()
        
        # Statistics for adaptive optimization
        self.loading_stats = LoadingStatistics()
            
    def load_dataset(
        self,
        mode: str = 'train',
        indices: Optional[List[int]] = None,
        transform_config: Optional[Dict[str, Any]] = None
    ) -> Union[Dataset, IterableDataset]:
        """Load dataset with mode-specific optimizations."""
        if self.streaming_mode:
            return StreamingBlockchainDataset(
                self.data_accessor,
                mode=mode,
                transform=self._get_transform(mode, transform_config),
                stream_processor=self.stream_processor
            )
        else:
            return BlockchainDataset(
                self.data_accessor,
                mode=mode,
                indices=indices,
                cache_manager=self.cache_manager,
                transform=self._get_transform(mode, transform_config),
                gpu_cache=self.gpu_cache
            )
    
    def create_distributed_loader(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = True
    ) -> DataLoader:
        """Create data loader optimized for distributed training."""
        # Create distributed sampler
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=self.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        
        # Determine optimal number of workers per GPU
        workers_per_gpu = max(1, self.num_workers // world_size)
        
        # Create data loader with optimizations
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=workers_per_gpu,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_batches,
            persistent_workers=True,
            collate_fn=partial(self._optimized_collate, rank=rank)
        )
        
        return loader
    
    def _optimized_collate(self, batch: List[Data], rank: int = 0) -> TransactionBatch:
        """Optimized collation function leveraging large memory and parallel processing."""
        batch_size = len(batch)
        
        # Use memory pool for allocation
        with self.memory_pool.allocate_context(estimated_size_gb=0.1 * batch_size):
            # Process samples in parallel
            with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
                # Submit processing tasks
                future_to_idx = {
                    executor.submit(self._process_sample, i, data, rank): i
                    for i, data in enumerate(batch)
                }
                
                # Collect results maintaining order
                processed_data = [None] * batch_size
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        processed_data[idx] = future.result()
                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        # Use fallback processing
                        processed_data[idx] = self._fallback_process_sample(batch[idx])
            
            # Aggregate processed data
            aggregated_batch = self._aggregate_batch(processed_data)
            
            # Update loading statistics
            self.loading_stats.update(batch_size, time.time())
            
            return aggregated_batch
    
    def _process_sample(self, idx: int, data: Data, rank: int) -> Dict[str, Any]:
        """Process individual sample with caching and error handling."""
        try:
            # Generate cache key
            cache_key = f"sample_{data.idx}_{rank}" if hasattr(data, 'idx') else f"sample_{idx}_{rank}"
            
            # Check cache
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                return cached
            
            # Extract sequences
            sequences = self._extract_sequences(data)
            
            # Process sample
            processed = {
                'idx': idx,
                'node_features': data.x if hasattr(data, 'x') else torch.zeros((1, self.feature_dim)),
                'edge_index': data.edge_index if hasattr(data, 'edge_index') else torch.zeros((2, 0), dtype=torch.long),
                'edge_attr': data.edge_attr if hasattr(data, 'edge_attr') else torch.zeros((0, self.metadata.get('edge_attr_dim', 8))),
                'sequences': sequences,
                'label': data.y if hasattr(data, 'y') else torch.zeros(1),
                'timestamp': data.timestamp if hasattr(data, 'timestamp') else torch.tensor(time.time()),
                'chain_id': data.chain_id if hasattr(data, 'chain_id') else 'unknown',
                'num_nodes': data.num_nodes if hasattr(data, 'num_nodes') else 1
            }
            
            # Cache processed data
            self.cache_manager.put(cache_key, processed, priority=1.0 if idx < 1000 else 0.5)
            
            return processed
            
        except Exception as e:
            print(f"Error in _process_sample: {e}")
            return self._fallback_process_sample(data)
    
    def _fallback_process_sample(self, data: Data) -> Dict[str, Any]:
        """Fallback processing for error cases."""
        return {
            'idx': 0,
            'node_features': torch.zeros((1, self.feature_dim)),
            'edge_index': torch.zeros((2, 0), dtype=torch.long),
            'edge_attr': torch.zeros((0, self.metadata.get('edge_attr_dim', 8))),
            'sequences': {
                'in_sequences': torch.zeros((32, 8)),
                'out_sequences': torch.zeros((32, 8)),
                'in_lengths': torch.ones(1, dtype=torch.long) * 32,
                'out_lengths': torch.ones(1, dtype=torch.long) * 32
            },
            'label': torch.zeros(1),
            'timestamp': torch.tensor(time.time()),
            'chain_id': 'unknown',
            'num_nodes': 1
        }
    
    def _extract_sequences(self, data: Data) -> Dict[str, torch.Tensor]:
        """Extract transaction sequences from graph data."""
        max_seq_len = self.metadata.get('max_sequence_length', 32)
        seq_dim = self.metadata.get('sequence_dim', 8)
        
        # Check if sequences are already in data
        if hasattr(data, 'in_sequences') and hasattr(data, 'out_sequences'):
            return {
                'in_sequences': data.in_sequences[:max_seq_len],
                'out_sequences': data.out_sequences[:max_seq_len],
                'in_lengths': data.in_lengths if hasattr(data, 'in_lengths') else torch.tensor([len(data.in_sequences)]),
                'out_lengths': data.out_lengths if hasattr(data, 'out_lengths') else torch.tensor([len(data.out_sequences)])
            }
        
        # Generate sequences from edge data
        if hasattr(data, 'edge_index') and data.edge_index.size(1) > 0:
            # Group edges by source and destination
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else torch.randn(edge_index.size(1), seq_dim)
            
            # Create sequences for each node
            num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else edge_index.max().item() + 1
            in_sequences = torch.zeros((num_nodes, max_seq_len, seq_dim))
            out_sequences = torch.zeros((num_nodes, max_seq_len, seq_dim))
            in_lengths = torch.zeros(num_nodes, dtype=torch.long)
            out_lengths = torch.zeros(num_nodes, dtype=torch.long)
            
            # Fill sequences
            for i, (src, dst) in enumerate(edge_index.t()):
                src, dst = src.item(), dst.item()
                
                # Out sequence for source
                if out_lengths[src] < max_seq_len:
                    out_sequences[src, out_lengths[src]] = edge_attr[i]
                    out_lengths[src] += 1
                
                # In sequence for destination
                if in_lengths[dst] < max_seq_len:
                    in_sequences[dst, in_lengths[dst]] = edge_attr[i]
                    in_lengths[dst] += 1
            
            # Average over nodes
            in_sequences = in_sequences.mean(dim=0)
            out_sequences = out_sequences.mean(dim=0)
            in_lengths = torch.tensor([in_lengths.float().mean().item()], dtype=torch.long)
            out_lengths = torch.tensor([out_lengths.float().mean().item()], dtype=torch.long)
        else:
            # Default sequences
            in_sequences = torch.zeros((max_seq_len, seq_dim))
            out_sequences = torch.zeros((max_seq_len, seq_dim))
            in_lengths = torch.tensor([1], dtype=torch.long)
            out_lengths = torch.tensor([1], dtype=torch.long)
        
        return {
            'in_sequences': in_sequences,
            'out_sequences': out_sequences,
            'in_lengths': in_lengths,
            'out_lengths': out_lengths
        }
    
    def _aggregate_batch(self, processed_data: List[Dict[str, Any]]) -> TransactionBatch:
        """Aggregate processed samples into a batch with proper padding and batching."""
        batch_size = len(processed_data)
        
        # Filter out None values
        processed_data = [d for d in processed_data if d is not None]
        if not processed_data:
            raise ValueError("No valid samples in batch")
        
        # Determine dimensions
        max_nodes = max(d['num_nodes'] for d in processed_data)
        max_edges = max(d['edge_index'].size(1) for d in processed_data)
        seq_len = self.metadata.get('max_sequence_length', 32)
        seq_dim = processed_data[0]['sequences']['in_sequences'].size(-1)
        
        # Pre-allocate tensors
        batch_node_features = []
        batch_edge_index = []
        batch_edge_attr = []
        batch_in_sequences = torch.zeros((batch_size, seq_len, seq_dim))
        batch_out_sequences = torch.zeros((batch_size, seq_len, seq_dim))
        batch_in_lengths = torch.zeros((batch_size,), dtype=torch.long)
        batch_out_lengths = torch.zeros((batch_size,), dtype=torch.long)
        batch_labels = torch.zeros((batch_size,), dtype=torch.long)
        batch_timestamps = torch.zeros((batch_size,))
        batch_chain_ids = []
        
        # Track graph boundaries
        node_offset = 0
        batch_ptr = [0]
        node_batch = []
        
        # Fill batch tensors
        for i, data in enumerate(processed_data):
            # Node features
            node_feat = data['node_features']
            batch_node_features.append(node_feat)
            
            # Edge index with offset
            edge_index = data['edge_index']
            if edge_index.size(1) > 0:
                batch_edge_index.append(edge_index + node_offset)
                batch_edge_attr.append(data['edge_attr'])
            
            # Update offset and batch assignment
            num_nodes = data['num_nodes']
            node_offset += num_nodes
            batch_ptr.append(node_offset)
            node_batch.extend([i] * num_nodes)
            
            # Sequences
            seq_data = data['sequences']
            seq_in = seq_data['in_sequences']
            seq_out = seq_data['out_sequences']
            len_in = min(seq_in.size(0), seq_len)
            len_out = min(seq_out.size(0), seq_len)
            
            batch_in_sequences[i, :len_in] = seq_in[:len_in]
            batch_out_sequences[i, :len_out] = seq_out[:len_out]
            batch_in_lengths[i] = seq_data['in_lengths'][0]
            batch_out_lengths[i] = seq_data['out_lengths'][0]
            
            # Other attributes
            batch_labels[i] = data['label']
            batch_timestamps[i] = data['timestamp']
            batch_chain_ids.append(data['chain_id'])
        
        # Concatenate node features and edges
        if batch_node_features:
            batch_node_features = torch.cat(batch_node_features, dim=0)
        else:
            batch_node_features = torch.zeros((1, self.feature_dim))
        
        if batch_edge_index:
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            batch_edge_attr = torch.cat(batch_edge_attr, dim=0)
        else:
            batch_edge_index = torch.zeros((2, 0), dtype=torch.long)
            batch_edge_attr = torch.zeros((0, self.metadata.get('edge_attr_dim', 8)))
        
        # Create batch object
        return TransactionBatch(
            node_features=batch_node_features,
            edge_index=batch_edge_index,
            edge_attr=batch_edge_attr,
            in_sequences=batch_in_sequences,
            out_sequences=batch_out_sequences,
            sequence_lengths={
                'in': batch_in_lengths,
                'out': batch_out_lengths
            },
            labels=batch_labels,
            timestamps=batch_timestamps,
            chain_ids=batch_chain_ids,
            batch_ptr=torch.tensor(batch_ptr, dtype=torch.long),
            node_batch=torch.tensor(node_batch, dtype=torch.long),
            metadata={
                'batch_size': batch_size,
                'total_nodes': node_offset,
                'total_edges': batch_edge_index.size(1)
            }
        )
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata with validation."""
        metadata_path = self.data_path / 'metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate required fields
                required_fields = ['num_samples', 'feature_dim']
                for field in required_fields:
                    if field not in metadata:
                        raise ValueError(f"Missing required field: {field}")
                
                return metadata
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return self._infer_metadata()
        else:
            return self._infer_metadata()
    
    def _infer_metadata(self) -> Dict[str, Any]:
        """Infer metadata from data files."""
        print("Inferring metadata from data files...")
        
        # Default metadata
        metadata = {
            'num_samples': 0,
            'feature_dim': 128,
            'edge_attr_dim': 8,
            'max_sequence_length': 32,
            'sequence_dim': 8,
            'num_chains': 1,
            'data_format': 'unknown'
        }
        
        # Try to infer from data files
        data_files = list(self.data_path.glob('*.pt'))
        if data_files:
            # Load first file to infer structure
            try:
                sample_data = torch.load(data_files[0], map_location='cpu')
                if isinstance(sample_data, Data):
                    if hasattr(sample_data, 'x'):
                        metadata['feature_dim'] = sample_data.x.size(-1)
                    if hasattr(sample_data, 'edge_attr'):
                        metadata['edge_attr_dim'] = sample_data.edge_attr.size(-1)
                
                metadata['num_samples'] = len(data_files)
                metadata['data_format'] = 'pytorch'
            except Exception as e:
                print(f"Error inferring metadata: {e}")
        
        # Check for HDF5 files
        hdf5_files = list(self.data_path.glob('*.h5'))
        if hdf5_files:
            try:
                with h5py.File(hdf5_files[0], 'r') as f:
                    if 'num_samples' in f.attrs:
                        metadata['num_samples'] = f.attrs['num_samples']
                    if 'features' in f:
                        metadata['feature_dim'] = f['features'].shape[-1]
                metadata['data_format'] = 'hdf5'
            except Exception as e:
                print(f"Error reading HDF5: {e}")
        
        return metadata
    
    def _get_transform(self, mode: str, config: Optional[Dict[str, Any]] = None) -> Optional[Callable]:
        """Get mode-specific data transformations."""
        transforms = []
        
        # Default configurations
        default_config = {
            'noise_std': 0.01,
            'edge_drop_p': 0.1,
            'node_drop_p': 0.05,
            'feature_noise': True,
            'augment_sequences': True
        }
        
        if config:
            default_config.update(config)
        
        if mode == 'train':
            # Training augmentations
            if default_config['feature_noise']:
                transforms.append(AddGaussianNoise(std=default_config['noise_std']))
            
            if default_config['edge_drop_p'] > 0:
                transforms.append(RandomEdgeDrop(p=default_config['edge_drop_p']))
            
            if default_config['node_drop_p'] > 0:
                transforms.append(RandomNodeDrop(p=default_config['node_drop_p']))
            
            if default_config['augment_sequences']:
                transforms.append(AugmentSequences())
        
        # Always normalize
        transforms.append(NormalizeFeatures())
        
        return compose_transforms(transforms) if transforms else None


class BlockchainDataset(Dataset):
    """
    Efficient dataset implementation for blockchain transaction data.
    Optimized for large-scale data with intelligent caching and prefetching.
    """
    
    def __init__(
        self,
        data_accessor: 'DataAccessor',
        mode: str = 'train',
        indices: Optional[List[int]] = None,
        cache_manager: Optional['CacheManager'] = None,
        transform: Optional[Callable] = None,
        gpu_cache: Optional['GPUCache'] = None,
        enable_prefetch: bool = True
    ):
        self.data_accessor = data_accessor
        self.mode = mode
        self.transform = transform
        self.cache_manager = cache_manager or CacheManager(max_size_gb=10)
        self.gpu_cache = gpu_cache
        self.enable_prefetch = enable_prefetch
        
        # Load indices
        if indices is not None:
            self.indices = indices
        else:
            self.indices = self._load_split_indices(mode)
        
        # Precompute access patterns for optimization
        self.access_pattern = self._analyze_access_pattern()
        
        # Initialize prefetch buffer
        if enable_prefetch:
            self.prefetch_buffer = PrefetchBuffer(
                window_size=self.access_pattern['prefetch_window']
            )
        
        # Track access statistics
        self.access_stats = defaultdict(int)
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Data:
        """Get item with multi-level caching and prefetching."""
        if idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.indices)}")
        
        real_idx = self.indices[idx]
        
        # Update access statistics
        self.access_stats[real_idx] += 1
        
        # Check GPU cache first (fastest)
        if self.gpu_cache is not None:
            gpu_cached = self.gpu_cache.get(f"{self.mode}_{real_idx}")
            if gpu_cached is not None:
                return gpu_cached
        
        # Check memory cache
        cache_key = f"{self.mode}_{real_idx}"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            # Move to GPU cache if frequently accessed
            if self.gpu_cache is not None and self.access_stats[real_idx] > 5:
                self.gpu_cache.put(cache_key, cached_data)
            return cached_data
        
        # Check prefetch buffer
        if self.enable_prefetch:
            prefetched = self.prefetch_buffer.get(real_idx)
            if prefetched is not None:
                data = prefetched
            else:
                # Load from disk
                data = self.data_accessor.load_sample(real_idx)
        else:
            # Load from disk
            data = self.data_accessor.load_sample(real_idx)
        
        # Apply transformations
        if self.transform is not None:
            data = self.transform(data)
        
        # Cache the processed data
        priority = self._calculate_cache_priority(real_idx, idx)
        self.cache_manager.put(cache_key, data, priority=priority)
        
        # Prefetch next items based on access pattern
        if self.enable_prefetch:
            self._prefetch_next(idx)
        
        return data
    
    def _load_split_indices(self, mode: str) -> List[int]:
        """Load train/val/test split indices with validation."""
        split_file = self.data_accessor.data_path / f'{mode}_indices.npy'
        
        if split_file.exists():
            try:
                indices = np.load(split_file)
                print(f"Loaded {len(indices)} {mode} indices from {split_file}")
                return indices.tolist()
            except Exception as e:
                print(f"Error loading split indices: {e}")
        
        # Default split if file doesn't exist
        total_samples = self.data_accessor.num_samples
        print(f"Creating default {mode} split from {total_samples} samples")
        
        if mode == 'train':
            indices = list(range(0, int(0.7 * total_samples)))
        elif mode == 'val':
            indices = list(range(int(0.7 * total_samples), int(0.85 * total_samples)))
        else:  # test
            indices = list(range(int(0.85 * total_samples), total_samples))
        
        # Save for future use
        try:
            np.save(split_file, np.array(indices))
        except Exception as e:
            print(f"Warning: Could not save split indices: {e}")
        
        return indices
    
    def _analyze_access_pattern(self) -> Dict[str, Any]:
        """Analyze data access patterns for optimization."""
        # Determine access pattern based on mode and dataset characteristics
        if self.mode == 'train':
            # Random access pattern for training
            pattern_type = 'random'
            prefetch_window = 8
            cache_priority = 'frequency'
        else:
            # Sequential access for validation/test
            pattern_type = 'sequential'
            prefetch_window = 16
            cache_priority = 'lru'
        
        return {
            'type': pattern_type,
            'prefetch_window': prefetch_window,
            'cache_priority': cache_priority,
            'batch_locality': 0.8  # Probability of accessing nearby samples
        }
    
    def _calculate_cache_priority(self, real_idx: int, access_idx: int) -> float:
        """Calculate cache priority based on access patterns."""
        base_priority = 1.0
        
        # Increase priority for frequently accessed samples
        frequency_bonus = min(self.access_stats[real_idx] * 0.1, 0.5)
        
        # Increase priority for recent accesses
        recency_bonus = 0.3 if access_idx < 100 else 0.0
        
        # Special samples (e.g., hard examples) get higher priority
        if hasattr(self, 'hard_samples') and real_idx in self.hard_samples:
            hard_sample_bonus = 0.5
        else:
            hard_sample_bonus = 0.0
        
        return base_priority + frequency_bonus + recency_bonus + hard_sample_bonus
    
    def _prefetch_next(self, current_idx: int):
        """Intelligently prefetch next items based on access patterns."""
        if not self.enable_prefetch:
            return
        
        prefetch_window = self.access_pattern['prefetch_window']
        pattern_type = self.access_pattern['type']
        
        # Determine which indices to prefetch
        if pattern_type == 'sequential':
            # Prefetch next sequential items
            prefetch_indices = [
                current_idx + offset 
                for offset in range(1, prefetch_window + 1)
                if current_idx + offset < len(self.indices)
            ]
        else:  # random
            # Prefetch based on batch locality
            locality = self.access_pattern['batch_locality']
            num_local = int(prefetch_window * locality)
            num_random = prefetch_window - num_local
            
            # Local indices (nearby)
            local_indices = [
                current_idx + offset 
                for offset in range(1, num_local + 1)
                if current_idx + offset < len(self.indices)
            ]
            
            # Random indices
            random_indices = random.sample(
                range(len(self.indices)),
                min(num_random, len(self.indices))
            )
            
            prefetch_indices = local_indices + random_indices
        
        # Submit prefetch tasks
        for idx in prefetch_indices:
            real_idx = self.indices[idx]
            if not self.prefetch_buffer.contains(real_idx):
                self.prefetch_buffer.prefetch(
                    real_idx,
                    lambda ridx: self.data_accessor.load_sample(ridx)
                )


class StreamingBlockchainDataset(IterableDataset):
    """
    Streaming dataset for real-time transaction processing.
    Implements sliding windows and incremental updates.
    """
    
    def __init__(
        self,
        data_accessor: 'DataAccessor',
        mode: str = 'train',
        window_size: int = 1000,
        update_interval: int = 100,
        transform: Optional[Callable] = None,
        stream_processor: Optional['StreamProcessor'] = None,
        buffer_size: int = 10000
    ):
        self.data_accessor = data_accessor
        self.mode = mode
        self.window_size = window_size
        self.update_interval = update_interval
        self.transform = transform
        self.stream_processor = stream_processor or StreamProcessor()
        self.buffer_size = buffer_size
        
        # Streaming components
        self.transaction_buffer = deque(maxlen=window_size)
        self.feature_buffer = StreamingFeatureBuffer(window_size)
        self.graph_builder = IncrementalGraphBuilder()
        
        # State tracking
        self.stream_position = 0
        self.graphs_generated = 0
        self.last_update_time = time.time()
        
    def __iter__(self) -> Iterator[Data]:
        """Iterate over streaming data with sliding windows."""
        # Initialize stream position based on mode
        if self.mode == 'train':
            start_pos = 0
            end_pos = int(0.7 * self.data_accessor.num_samples)
        elif self.mode == 'val':
            start_pos = int(0.7 * self.data_accessor.num_samples)
            end_pos = int(0.85 * self.data_accessor.num_samples)
        else:  # test
            start_pos = int(0.85 * self.data_accessor.num_samples)
            end_pos = self.data_accessor.num_samples
        
        self.stream_position = start_pos
        
        # Main streaming loop
        while self.stream_position < end_pos:
            # Get batch of transactions
            batch_size = min(self.update_interval, end_pos - self.stream_position)
            transactions = []
            
            for _ in range(batch_size):
                if self.stream_position >= end_pos:
                    break
                
                transaction = self._get_next_transaction()
                if transaction is not None:
                    transactions.append(transaction)
                    self.stream_position += 1
            
            if not transactions:
                break
            
            # Process batch of transactions
            for transaction in transactions:
                # Update buffers
                self.transaction_buffer.append(transaction)
                self.feature_buffer.update(transaction)
                
                # Process with stream processor
                if self.stream_processor:
                    processed = self.stream_processor.process_transaction(
                        transaction['data'],
                        transaction['timestamp']
                    )
                    if processed is not None:
                        yield processed
            
            # Create graph snapshot at intervals
            if len(self.transaction_buffer) >= self.window_size and self.graphs_generated % self.update_interval == 0:
                graph_data = self._create_graph_snapshot()
                
                # Apply transformations
                if self.transform is not None:
                    graph_data = self.transform(graph_data)
                
                self.graphs_generated += 1
                yield graph_data
            
            # Periodic maintenance
            if time.time() - self.last_update_time > 60:  # Every minute
                self._perform_maintenance()
                self.last_update_time = time.time()
    
    def _get_next_transaction(self) -> Optional[Dict[str, Any]]:
        """Get next transaction from stream with error handling."""
        try:
            # Load transaction data
            tx_data = self.data_accessor.load_transaction(self.stream_position)
            
            if tx_data is None:
                return None
            
            # Ensure required fields
            if 'source' not in tx_data or 'destination' not in tx_data:
                print(f"Warning: Invalid transaction at position {self.stream_position}")
                return None
            
            # Add metadata
            return {
                'data': tx_data,
                'timestamp': tx_data.get('timestamp', time.time()),
                'position': self.stream_position
            }
            
        except Exception as e:
            print(f"Error loading transaction at position {self.stream_position}: {e}")
            return None
    
    def _create_graph_snapshot(self) -> Data:
        """Create graph from current window with proper feature extraction."""
        # Get current transactions
        transactions = list(self.transaction_buffer)
        
        if not transactions:
            # Return empty graph
            return Data(
                x=torch.zeros((1, self.data_accessor.metadata.get('feature_dim', 128))),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, self.data_accessor.metadata.get('edge_attr_dim', 8)))
            )
        
        # Build graph structure
        graph_dict = self.graph_builder.build_from_transactions(transactions)
        
        # Extract node features
        node_features = []
        for node_id in graph_dict['nodes']:
            features = self.feature_buffer.get_node_features(node_id)
            node_features.append(features)
        
        node_features = torch.stack(node_features) if node_features else torch.zeros((1, 128))
        
        # Extract edge features
        edge_features = []
        edge_index = graph_dict['edge_index']
        
        for i in range(edge_index.size(1)):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()
            src_node = graph_dict['nodes'][src_idx]
            dst_node = graph_dict['nodes'][dst_idx]
            
            # Find corresponding transaction
            edge_feat = None
            for tx_info in transactions:
                tx = tx_info['data']
                if tx['source'] == src_node and tx['destination'] == dst_node:
                    edge_feat = self._extract_edge_features(tx)
                    break
            
            if edge_feat is None:
                edge_feat = torch.zeros(self.data_accessor.metadata.get('edge_attr_dim', 8))
            
            edge_features.append(edge_feat)
        
        edge_features = torch.stack(edge_features) if edge_features else torch.zeros((0, 8))
        
        # Add global graph features
        window_stats = self.feature_buffer.get_window_statistics()
        
        # Create PyG data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            num_nodes=len(graph_dict['nodes']),
            window_stats=window_stats,
            timestamp=torch.tensor(time.time())
        )
        
        return data
    
    def _extract_edge_features(self, transaction: Dict[str, Any]) -> torch.Tensor:
        """Extract features from a transaction for edge attributes."""
        features = []
        
        # Amount (log-scaled)
        amount = transaction.get('amount', 0.0)
        features.append(np.log1p(amount))
        
        # Time features
        timestamp = transaction.get('timestamp', time.time())
        hour = (timestamp % 86400) / 3600
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Transaction type features
        tx_type = transaction.get('type', 'unknown')
        type_features = [0.0] * 5  # 5 transaction types
        type_map = {'transfer': 0, 'contract': 1, 'mining': 2, 'exchange': 3, 'other': 4}
        if tx_type in type_map:
            type_features[type_map[tx_type]] = 1.0
        features.extend(type_features)
        
        # Ensure correct dimension
        while len(features) < self.data_accessor.metadata.get('edge_attr_dim', 8):
            features.append(0.0)
        
        return torch.tensor(features[:self.data_accessor.metadata.get('edge_attr_dim', 8)], dtype=torch.float32)
    
    def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        # Clear old data from feature buffer
        self.feature_buffer.cleanup_old_data()
        
        # Garbage collection
        gc.collect()
        
        # Log statistics
        print(f"Streaming dataset: processed {self.stream_position} transactions, "
              f"generated {self.graphs_generated} graphs")


class NetworkTopologyGenerator:
    """
    Generates various network topologies for simulation and testing.
    Implements realistic blockchain network structures.
    """
    
    @staticmethod
    def generate(
        topology_type: str,
        num_nodes: int,
        seed: Optional[int] = None,
        **kwargs
    ) -> nx.Graph:
        """Generate network topology of specified type with reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        generators = {
            'scale_free': NetworkTopologyGenerator._generate_scale_free,
            'small_world': NetworkTopologyGenerator._generate_small_world,
            'random': NetworkTopologyGenerator._generate_random,
            'hierarchical': NetworkTopologyGenerator._generate_hierarchical,
            'blockchain': NetworkTopologyGenerator._generate_blockchain_network,
            'hybrid': NetworkTopologyGenerator._generate_hybrid,
            'geographic': NetworkTopologyGenerator._generate_geographic,
            'dynamic': NetworkTopologyGenerator._generate_dynamic
        }
        
        if topology_type not in generators:
            raise ValueError(f"Unknown topology type: {topology_type}. Available: {list(generators.keys())}")
        
        G = generators[topology_type](num_nodes, **kwargs)
        
        # Add metadata
        G.graph['topology_type'] = topology_type
        G.graph['num_nodes'] = num_nodes
        G.graph['generated_at'] = time.time()
        
        return G
    
    @staticmethod
    def _generate_scale_free(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate scale-free network (Barabási-Albert model)."""
        m = kwargs.get('m', 3)  # Number of edges to attach
        seed_graph = kwargs.get('seed_graph', None)
        
        G = nx.barabasi_albert_graph(num_nodes, m, seed=seed_graph)
        
        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['degree_centrality'] = G.degree(node) / (num_nodes - 1)
            G.nodes[node]['node_type'] = 'hub' if G.degree(node) > 2 * m else 'regular'
        
        return G
    
    @staticmethod
    def _generate_small_world(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate small-world network (Watts-Strogatz model)."""
        k = kwargs.get('k', 6)  # Each node connected to k nearest neighbors
        p = kwargs.get('p', 0.3)  # Rewiring probability
        
        G = nx.watts_strogatz_graph(num_nodes, k, p)
        
        # Calculate clustering coefficient for each node
        clustering = nx.clustering(G)
        for node in G.nodes():
            G.nodes[node]['clustering_coefficient'] = clustering[node]
            G.nodes[node]['node_type'] = 'bridge' if clustering[node] < 0.3 else 'cluster'
        
        return G
    
    @staticmethod
    def _generate_random(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate random network (Erdős-Rényi model)."""
        p = kwargs.get('p', 0.1)  # Edge probability
        directed = kwargs.get('directed', False)
        
        if directed:
            G = nx.erdos_renyi_graph(num_nodes, p, directed=True)
        else:
            G = nx.erdos_renyi_graph(num_nodes, p)
        
        # Add random weights
        for u, v in G.edges():
            G.edges[u, v]['weight'] = np.random.uniform(0.1, 1.0)
        
        return G
    
    @staticmethod
    def _generate_hierarchical(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate hierarchical network structure."""
        levels = kwargs.get('levels', 3)
        branching_factor = kwargs.get('branching_factor', 4)
        inter_level_prob = kwargs.get('inter_level_prob', 0.1)
        
        G = nx.Graph()
        
        # Create hierarchical structure
        node_id = 0
        level_nodes = {0: []}
        
        # Root level
        G.add_node(node_id, level=0, role='root')
        level_nodes[0].append(node_id)
        node_id += 1
        
        # Build hierarchy
        for level in range(1, levels):
            level_nodes[level] = []
            
            # Determine number of nodes at this level
            parent_count = len(level_nodes[level - 1])
            nodes_at_level = min(
                parent_count * branching_factor,
                num_nodes - node_id
            )
            
            if nodes_at_level == 0:
                break
            
            # Distribute nodes among parents
            nodes_per_parent = nodes_at_level // parent_count
            extra_nodes = nodes_at_level % parent_count
            
            for i, parent in enumerate(level_nodes[level - 1]):
                # Number of children for this parent
                num_children = nodes_per_parent + (1 if i < extra_nodes else 0)
                
                for _ in range(num_children):
                    if node_id >= num_nodes:
                        break
                    
                    role = 'leaf' if level == levels - 1 else 'intermediate'
                    G.add_node(node_id, level=level, role=role)
                    G.add_edge(parent, node_id, edge_type='hierarchical')
                    level_nodes[level].append(node_id)
                    node_id += 1
        
        # Add cross-level connections
        for level in range(1, len(level_nodes)):
            nodes_in_level = level_nodes[level]
            if len(nodes_in_level) > 1:
                num_cross_edges = int(len(nodes_in_level) * inter_level_prob)
                
                for _ in range(num_cross_edges):
                    u, v = random.sample(nodes_in_level, 2)
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, edge_type='lateral')
        
        # Fill remaining nodes if needed
        while node_id < num_nodes:
            # Attach to random existing node
            parent = random.choice(list(G.nodes()))
            G.add_node(node_id, level=-1, role='auxiliary')
            G.add_edge(parent, node_id, edge_type='auxiliary')
            node_id += 1
        
        return G
    
    @staticmethod
    def _generate_blockchain_network(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate realistic blockchain network topology."""
        # Network composition parameters
        num_miners = kwargs.get('num_miners', max(10, int(num_nodes * 0.05)))
        num_full_nodes = kwargs.get('num_full_nodes', max(50, int(num_nodes * 0.15)))
        num_exchanges = kwargs.get('num_exchanges', max(5, int(num_nodes * 0.02)))
        
        # Ensure we don't exceed total nodes
        num_miners = min(num_miners, num_nodes // 4)
        num_full_nodes = min(num_full_nodes, (num_nodes - num_miners) // 2)
        num_exchanges = min(num_exchanges, (num_nodes - num_miners - num_full_nodes) // 4)
        num_light_nodes = num_nodes - num_miners - num_full_nodes - num_exchanges
        
        G = nx.Graph()
        node_id = 0
        
        # Create node groups
        miners = []
        full_nodes = []
        exchanges = []
        light_nodes = []
        
        # Add miners (highly connected, high capacity)
        for _ in range(num_miners):
            G.add_node(node_id, 
                      node_type='miner',
                      capacity=1.0,
                      processing_power=random.uniform(0.8, 1.0),
                      latency=random.uniform(10, 30))
            miners.append(node_id)
            node_id += 1
        
        # Add full nodes (well connected, medium capacity)
        for _ in range(num_full_nodes):
            G.add_node(node_id,
                      node_type='full',
                      capacity=0.7,
                      processing_power=random.uniform(0.5, 0.8),
                      latency=random.uniform(20, 50))
            full_nodes.append(node_id)
            node_id += 1
        
        # Add exchanges (highly connected to light nodes)
        for _ in range(num_exchanges):
            G.add_node(node_id,
                      node_type='exchange',
                      capacity=0.9,
                      processing_power=random.uniform(0.7, 0.9),
                      latency=random.uniform(5, 20))
            exchanges.append(node_id)
            node_id += 1
        
        # Add light nodes (sparsely connected, low capacity)
        for _ in range(num_light_nodes):
            G.add_node(node_id,
                      node_type='light',
                      capacity=0.3,
                      processing_power=random.uniform(0.1, 0.5),
                      latency=random.uniform(30, 100))
            light_nodes.append(node_id)
            node_id += 1
        
        # Connect network components
        
        # 1. Miners form a nearly complete graph (mining pool)
        for i in range(len(miners)):
            for j in range(i + 1, len(miners)):
                if random.random() < 0.8:  # 80% connection probability
                    G.add_edge(miners[i], miners[j],
                             weight=1.0,
                             bandwidth=random.uniform(100, 1000),
                             edge_type='miner-miner')
        
        # 2. Full nodes connect to miners and each other
        for full_node in full_nodes:
            # Connect to 3-5 random miners
            num_miner_connections = random.randint(3, min(5, len(miners)))
            connected_miners = random.sample(miners, num_miner_connections)
            for miner in connected_miners:
                G.add_edge(full_node, miner,
                         weight=0.8,
                         bandwidth=random.uniform(50, 500),
                         edge_type='full-miner')
            
            # Connect to 2-4 other full nodes (small world)
            num_full_connections = random.randint(2, min(4, len(full_nodes) - 1))
            potential_connections = [fn for fn in full_nodes if fn != full_node]
            if potential_connections:
                connected_full = random.sample(potential_connections, 
                                             min(num_full_connections, len(potential_connections)))
                for other_full in connected_full:
                    if not G.has_edge(full_node, other_full):
                        G.add_edge(full_node, other_full,
                                 weight=0.7,
                                 bandwidth=random.uniform(30, 300),
                                 edge_type='full-full')
        
        # 3. Exchanges connect to multiple full nodes and some miners
        for exchange in exchanges:
            # Connect to 5-10 full nodes
            num_full_connections = random.randint(5, min(10, len(full_nodes)))
            connected_full = random.sample(full_nodes, num_full_connections)
            for full_node in connected_full:
                G.add_edge(exchange, full_node,
                         weight=0.9,
                         bandwidth=random.uniform(100, 1000),
                         edge_type='exchange-full')
            
            # Connect to 1-3 miners directly
            num_miner_connections = random.randint(1, min(3, len(miners)))
            connected_miners = random.sample(miners, num_miner_connections)
            for miner in connected_miners:
                G.add_edge(exchange, miner,
                         weight=0.85,
                         bandwidth=random.uniform(200, 2000),
                         edge_type='exchange-miner')
        
        # 4. Light nodes connect to full nodes and exchanges
        for light_node in light_nodes:
            # Prefer connection to exchanges (user wallets)
            if exchanges and random.random() < 0.7:
                exchange = random.choice(exchanges)
                G.add_edge(light_node, exchange,
                         weight=0.6,
                         bandwidth=random.uniform(10, 100),
                         edge_type='light-exchange')
            
            # Connect to 1-3 full nodes
            num_full_connections = random.randint(1, min(3, len(full_nodes)))
            connected_full = random.sample(full_nodes, num_full_connections)
            for full_node in connected_full:
                G.add_edge(light_node, full_node,
                         weight=0.5,
                         bandwidth=random.uniform(10, 50),
                         edge_type='light-full')
        
        # Add geographic clustering
        if kwargs.get('geographic', False):
            NetworkTopologyGenerator._add_geographic_clustering(G)
        
        return G
    
    @staticmethod
    def _generate_hybrid(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate hybrid topology combining multiple models."""
        # Split nodes between different topology types
        component_sizes = kwargs.get('component_sizes', [0.3, 0.3, 0.4])
        component_types = kwargs.get('component_types', ['scale_free', 'small_world', 'blockchain'])
        
        if len(component_sizes) != len(component_types):
            raise ValueError("component_sizes and component_types must have same length")
        
        # Normalize sizes
        total = sum(component_sizes)
        component_sizes = [s / total for s in component_sizes]
        
        # Create components
        components = []
        node_count = 0
        
        for size, comp_type in zip(component_sizes, component_types):
            comp_nodes = int(num_nodes * size)
            if comp_nodes > 0:
                # Generate component
                comp_graph = NetworkTopologyGenerator.generate(
                    comp_type, comp_nodes, **kwargs
                )
                
                # Relabel nodes to avoid conflicts
                mapping = {old: old + node_count for old in comp_graph.nodes()}
                comp_graph = nx.relabel_nodes(comp_graph, mapping)
                
                components.append(comp_graph)
                node_count += comp_nodes
        
        # Combine components
        G = nx.Graph()
        for comp in components:
            G = nx.compose(G, comp)
        
        # Add inter-component edges
        num_bridges = kwargs.get('num_bridges', int(np.sqrt(num_nodes)))
        bridge_strategy = kwargs.get('bridge_strategy', 'random')
        
        for _ in range(num_bridges):
            # Select two different components
            if len(components) >= 2:
                comp1, comp2 = random.sample(components, 2)
                
                if bridge_strategy == 'random':
                    # Random connection
                    node1 = random.choice(list(comp1.nodes()))
                    node2 = random.choice(list(comp2.nodes()))
                elif bridge_strategy == 'hub':
                    # Connect high-degree nodes
                    node1 = max(comp1.nodes(), key=lambda n: comp1.degree(n))
                    node2 = max(comp2.nodes(), key=lambda n: comp2.degree(n))
                else:
                    # Default to random
                    node1 = random.choice(list(comp1.nodes()))
                    node2 = random.choice(list(comp2.nodes()))
                
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2,
                             weight=0.5,
                             edge_type='bridge',
                             bandwidth=random.uniform(50, 200))
        
        return G
    
    @staticmethod
    def _generate_geographic(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate geographically distributed network."""
        # Parameters
        num_regions = kwargs.get('num_regions', int(np.sqrt(num_nodes)))
        intra_region_prob = kwargs.get('intra_region_prob', 0.8)
        inter_region_prob = kwargs.get('inter_region_prob', 0.1)
        
        G = nx.Graph()
        
        # Assign nodes to regions
        nodes_per_region = num_nodes // num_regions
        extra_nodes = num_nodes % num_regions
        
        node_id = 0
        region_nodes = defaultdict(list)
        
        for region in range(num_regions):
            # Region center coordinates
            region_x = (region % int(np.sqrt(num_regions))) * 100
            region_y = (region // int(np.sqrt(num_regions))) * 100
            
            # Number of nodes in this region
            region_size = nodes_per_region + (1 if region < extra_nodes else 0)
            
            for _ in range(region_size):
                # Add node with geographic coordinates
                x = region_x + random.gauss(0, 20)
                y = region_y + random.gauss(0, 20)
                
                G.add_node(node_id,
                         region=region,
                         x=x,
                         y=y,
                         latency_factor=random.uniform(0.8, 1.2))
                
                region_nodes[region].append(node_id)
                node_id += 1
        
        # Connect nodes within regions
        for region, nodes in region_nodes.items():
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    if random.random() < intra_region_prob:
                        # Distance-based weight
                        dist = np.sqrt(
                            (G.nodes[u]['x'] - G.nodes[v]['x'])**2 +
                            (G.nodes[u]['y'] - G.nodes[v]['y'])**2
                        )
                        weight = 1.0 / (1.0 + dist / 50)
                        
                        G.add_edge(u, v,
                                 weight=weight,
                                 distance=dist,
                                 edge_type='intra-region')
        
        # Connect nodes between regions
        regions = list(region_nodes.keys())
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                # Connect some nodes between regions
                num_inter_edges = int(len(region_nodes[region1]) * len(region_nodes[region2]) * inter_region_prob)
                
                for _ in range(max(1, num_inter_edges)):
                    u = random.choice(region_nodes[region1])
                    v = random.choice(region_nodes[region2])
                    
                    if not G.has_edge(u, v):
                        dist = np.sqrt(
                            (G.nodes[u]['x'] - G.nodes[v]['x'])**2 +
                            (G.nodes[u]['y'] - G.nodes[v]['y'])**2
                        )
                        weight = 0.5 / (1.0 + dist / 100)
                        
                        G.add_edge(u, v,
                                 weight=weight,
                                 distance=dist,
                                 edge_type='inter-region')
        
        return G
    
    @staticmethod
    def _generate_dynamic(num_nodes: int, **kwargs) -> nx.Graph:
        """Generate network with dynamic properties for temporal analysis."""
        # Start with a base topology
        base_type = kwargs.get('base_type', 'scale_free')
        G = NetworkTopologyGenerator.generate(base_type, num_nodes, **kwargs)
        
        # Add temporal properties
        time_periods = kwargs.get('time_periods', 10)
        change_rate = kwargs.get('change_rate', 0.1)
        
        # Store evolution history
        G.graph['evolution_history'] = []
        
        for period in range(time_periods):
            # Record current state
            snapshot = {
                'period': period,
                'num_edges': G.number_of_edges(),
                'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
                'clustering': nx.average_clustering(G)
            }
            G.graph['evolution_history'].append(snapshot)
            
            # Apply changes
            num_changes = int(G.number_of_edges() * change_rate)
            
            # Remove some edges
            edges_to_remove = random.sample(list(G.edges()), num_changes // 2)
            G.remove_edges_from(edges_to_remove)
            
            # Add new edges
            for _ in range(num_changes // 2):
                u, v = random.sample(list(G.nodes()), 2)
                if not G.has_edge(u, v):
                    G.add_edge(u, v,
                             weight=random.uniform(0.1, 1.0),
                             created_at=period)
        
        # Mark nodes with activity patterns
        for node in G.nodes():
            G.nodes[node]['activity_pattern'] = random.choice(['constant', 'periodic', 'bursty', 'declining'])
            G.nodes[node]['activity_level'] = random.uniform(0.1, 1.0)
        
        return G
    
    @staticmethod
    def _add_geographic_clustering(G: nx.Graph):
        """Add geographic clustering to existing graph."""
        # Assign nodes to geographic regions
        num_regions = int(np.sqrt(G.number_of_nodes()))
        
        for node in G.nodes():
            region = node % num_regions
            G.nodes[node]['region'] = region
            
            # Add coordinates
            region_x = (region % int(np.sqrt(num_regions))) * 100
            region_y = (region // int(np.sqrt(num_regions))) * 100
            G.nodes[node]['x'] = region_x + random.gauss(0, 20)
            G.nodes[node]['y'] = region_y + random.gauss(0, 20)


class MetricsCalculator:
    """
    Calculates comprehensive metrics for blockchain anomaly detection
    with focus on networking performance indicators.
    """
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.metric_history = defaultdict(list)
        self.network_metrics = NetworkMetrics()
        self.detection_metrics = DetectionMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.custom_metrics = {}
        
        # For distributed training
        self.distributed = torch.distributed.is_initialized() if DISTRIBUTED_AVAILABLE else False
        
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        routing_decisions: Optional[Dict[str, torch.Tensor]] = None,
        network_state: Optional[Dict[str, float]] = None,
        timing_info: Optional[Dict[str, float]] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ):
        """Update all metrics with new batch results."""
        # Move tensors to CPU for metric calculation
        predictions = predictions.detach().cpu()
        labels = labels.detach().cpu()
        
        # Detection metrics
        detection_results = self.detection_metrics.compute(predictions, labels)
        self._update_history('detection', detection_results)
        
        # Network metrics
        if routing_decisions is not None and network_state is not None:
            # Move routing tensors to CPU
            routing_decisions_cpu = {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in routing_decisions.items()
            }
            network_results = self.network_metrics.compute(
                routing_decisions_cpu, network_state
            )
            self._update_history('network', network_results)
        
        # Efficiency metrics
        if timing_info is not None:
            efficiency_results = self.efficiency_metrics.compute(
                timing_info, len(predictions)
            )
            self._update_history('efficiency', efficiency_results)
        
        # Custom metrics
        if custom_metrics is not None:
            self._update_history('custom', custom_metrics)
    
    def compute_epoch_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics for epoch."""
        epoch_metrics = {}
        
        # Aggregate detection metrics
        if 'detection' in self.metric_history:
            detection_history = self.metric_history['detection']
            epoch_metrics.update({
                'accuracy': np.mean([m['accuracy'] for m in detection_history]),
                'precision': np.mean([m['precision'] for m in detection_history]),
                'recall': np.mean([m['recall'] for m in detection_history]),
                'f1_score': np.mean([m['f1_score'] for m in detection_history]),
                'auc': np.mean([m['auc'] for m in detection_history])
            })
            
            # Add confusion matrix totals
            tp = sum(m['true_positives'] for m in detection_history)
            fp = sum(m['false_positives'] for m in detection_history)
            tn = sum(m['true_negatives'] for m in detection_history)
            fn = sum(m['false_negatives'] for m in detection_history)
            
            epoch_metrics['confusion_matrix'] = {
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
            }
        
        # Aggregate network metrics
        if 'network' in self.metric_history:
            network_history = self.metric_history['network']
            epoch_metrics.update({
                'avg_latency': np.mean([m['latency'] for m in network_history]),
                'p95_latency': np.mean([m.get('latency_p95', m['latency']) for m in network_history]),
                'p99_latency': np.mean([m.get('latency_p99', m['latency']) for m in network_history]),
                'throughput': np.sum([m['throughput'] for m in network_history]),
                'packet_loss_rate': np.mean([m['packet_loss'] for m in network_history]),
                'routing_efficiency': np.mean([m['routing_efficiency'] for m in network_history]),
                'network_utilization': np.mean([m.get('network_utilization', 0) for m in network_history])
            })
        
        # Aggregate efficiency metrics
        if 'efficiency' in self.metric_history:
            efficiency_history = self.metric_history['efficiency']
            epoch_metrics.update({
                'avg_processing_time': np.mean([m['processing_time'] for m in efficiency_history]),
                'samples_per_second': np.mean([m['samples_per_second'] for m in efficiency_history]),
                'memory_usage': np.mean([m['memory_usage'] for m in efficiency_history]),
                'gpu_utilization': np.mean([m.get('gpu_utilization', 0) for m in efficiency_history]),
                'gpu_memory_usage': np.mean([m.get('gpu_memory_usage', 0) for m in efficiency_history])
            })
            
            # Communication overhead for distributed training
            if any('communication_overhead' in m for m in efficiency_history):
                epoch_metrics['communication_overhead'] = np.mean([
                    m.get('communication_overhead', 0) for m in efficiency_history
                ])
        
        # Custom metrics
        if 'custom' in self.metric_history:
            custom_history = self.metric_history['custom']
            for key in custom_history[0].keys():
                if isinstance(custom_history[0][key], (int, float)):
                    epoch_metrics[f'custom_{key}'] = np.mean([m[key] for m in custom_history])
        
        return epoch_metrics
    
    def synchronize_metrics(self) -> Dict[str, float]:
        """Synchronize metrics across distributed processes."""
        epoch_metrics = self.compute_epoch_metrics()
        
        if self.distributed:
            # Convert metrics to tensors for all_reduce
            metric_tensors = {}
            
            for key, value in epoch_metrics.items():
                if isinstance(value, (int, float)):
                    tensor = torch.tensor(value, device=self.device)
                    torch.distributed.all_reduce(tensor)
                    metric_tensors[key] = tensor.item() / torch.distributed.get_world_size()
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like confusion matrix)
                    metric_tensors[key] = {}
                    for sub_key, sub_value in value.items():
                        tensor = torch.tensor(sub_value, device=self.device)
                        torch.distributed.all_reduce(tensor)
                        metric_tensors[key][sub_key] = tensor.item()
                else:
                    metric_tensors[key] = value
            
            return metric_tensors
        else:
            return epoch_metrics
    
    def get_summary_string(self, metrics: Optional[Dict[str, float]] = None) -> str:
        """Get formatted summary string of metrics."""
        if metrics is None:
            metrics = self.compute_epoch_metrics()
        
        summary_parts = []
        
        # Detection metrics
        if 'accuracy' in metrics:
            summary_parts.append(f"Acc: {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            summary_parts.append(f"Prec: {metrics['precision']:.4f}")
        if 'recall' in metrics:
            summary_parts.append(f"Rec: {metrics['recall']:.4f}")
        if 'f1_score' in metrics:
            summary_parts.append(f"F1: {metrics['f1_score']:.4f}")
        if 'auc' in metrics:
            summary_parts.append(f"AUC: {metrics['auc']:.4f}")
        
        # Network metrics
        if 'avg_latency' in metrics:
            summary_parts.append(f"Lat: {metrics['avg_latency']:.1f}ms")
        if 'throughput' in metrics:
            summary_parts.append(f"Thr: {metrics['throughput']:.1f}tx/s")
        if 'routing_efficiency' in metrics:
            summary_parts.append(f"RoutEff: {metrics['routing_efficiency']:.3f}")
        
        # Efficiency metrics
        if 'samples_per_second' in metrics:
            summary_parts.append(f"Speed: {metrics['samples_per_second']:.1f}samp/s")
        
        return " | ".join(summary_parts)
    
    def save_metrics(self, filepath: Path):
        """Save metrics history to file."""
        metrics_data = {
            'history': dict(self.metric_history),
            'epoch_summary': self.compute_epoch_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self, filepath: Path):
        """Load metrics history from file."""
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)
        
        self.metric_history = defaultdict(list, metrics_data['history'])
    
    def reset(self):
        """Reset all metric histories."""
        self.metric_history.clear()
    
    def _update_history(self, category: str, metrics: Dict[str, Any]):
        """Update metric history with new values."""
        self.metric_history[category].append(metrics)


class NetworkMetrics:
    """Calculates network-specific performance metrics for INFOCOM evaluation."""
    
    def compute(
        self,
        routing_decisions: Dict[str, Any],
        network_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute comprehensive network performance metrics."""
        metrics = {}
        
        # Latency metrics
        if 'path_latencies' in routing_decisions:
            latencies = routing_decisions['path_latencies']
            if isinstance(latencies, torch.Tensor):
                latencies = latencies.numpy()
            elif not isinstance(latencies, np.ndarray):
                latencies = np.array(latencies)
            
            metrics['latency'] = float(np.mean(latencies))
            metrics['latency_std'] = float(np.std(latencies))
            metrics['latency_p50'] = float(np.percentile(latencies, 50))
            metrics['latency_p95'] = float(np.percentile(latencies, 95))
            metrics['latency_p99'] = float(np.percentile(latencies, 99))
            metrics['latency_max'] = float(np.max(latencies))
        
        # Throughput calculation
        if 'timestamps' in routing_decisions:
            timestamps = routing_decisions['timestamps']
            if isinstance(timestamps, torch.Tensor):
                timestamps = timestamps.numpy()
            
            if len(timestamps) > 1:
                time_window = float(timestamps[-1] - timestamps[0])
                num_transactions = len(timestamps)
                metrics['throughput'] = num_transactions / max(time_window, 1e-6)
                
                # Calculate throughput variance
                if len(timestamps) > 10:
                    # Sliding window throughput
                    window_size = min(10, len(timestamps) // 5)
                    throughputs = []
                    for i in range(len(timestamps) - window_size):
                        window_time = timestamps[i + window_size] - timestamps[i]
                        window_throughput = window_size / max(window_time, 1e-6)
                        throughputs.append(window_throughput)
                    
                    metrics['throughput_variance'] = float(np.var(throughputs))
            else:
                metrics['throughput'] = 0.0
        
        # Packet loss and reliability
        metrics['packet_loss'] = network_state.get('packet_loss_rate', 0.0)
        metrics['reliability'] = 1.0 - metrics['packet_loss']
        
        # Routing efficiency metrics
        if 'path_lengths' in routing_decisions and 'optimal_lengths' in routing_decisions:
            actual_lengths = routing_decisions['path_lengths']
            optimal_lengths = routing_decisions['optimal_lengths']
            
            if isinstance(actual_lengths, torch.Tensor):
                actual_lengths = actual_lengths.numpy()
            if isinstance(optimal_lengths, torch.Tensor):
                optimal_lengths = optimal_lengths.numpy()
            
            # Avoid division by zero
            efficiency_ratios = optimal_lengths / np.maximum(actual_lengths, 1)
            metrics['routing_efficiency'] = float(np.mean(efficiency_ratios))
            metrics['routing_efficiency_std'] = float(np.std(efficiency_ratios))
            
            # Path stretch factor
            stretch_factors = actual_lengths / np.maximum(optimal_lengths, 1)
            metrics['avg_path_stretch'] = float(np.mean(stretch_factors))
        
        # Network utilization and congestion
        metrics['network_utilization'] = network_state.get('utilization', 0.0)
        metrics['congestion_level'] = network_state.get('congestion_level', 0.0)
        
        # Bandwidth metrics
        if 'bandwidth_usage' in network_state:
            metrics['bandwidth_utilization'] = network_state['bandwidth_usage']
        
        # Jitter calculation
        if 'path_latencies' in routing_decisions and len(routing_decisions['path_latencies']) > 1:
            latency_diffs = np.diff(routing_decisions['path_latencies'])
            metrics['jitter'] = float(np.mean(np.abs(latency_diffs)))
        
        # QoS violations
        if 'qos_violations' in network_state:
            metrics['qos_violation_rate'] = network_state['qos_violations']
        
        # Load balancing metrics
        if 'node_loads' in network_state:
            loads = list(network_state['node_loads'].values())
            metrics['load_balance_factor'] = float(np.std(loads) / (np.mean(loads) + 1e-6))
        
        return metrics


class DetectionMetrics:
    """Calculates detection performance metrics with additional blockchain-specific metrics."""
    
    def compute(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Compute comprehensive detection metrics."""
        # Ensure tensors are on CPU
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # Handle different prediction formats
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class predictions
            pred_probs = predictions
            pred_classes = predictions.argmax(axis=1)
        else:
            # Binary predictions
            if len(predictions.shape) > 1:
                predictions = predictions.squeeze()
            pred_probs = predictions
            pred_classes = (predictions > 0.5).astype(int)
        
        # Basic metrics
        correct = (pred_classes == labels).sum()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0
        
        # Confusion matrix elements
        tp = ((pred_classes == 1) & (labels == 1)).sum()
        fp = ((pred_classes == 1) & (labels == 0)).sum()
        tn = ((pred_classes == 0) & (labels == 0)).sum()
        fn = ((pred_classes == 0) & (labels == 1)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity and NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # AUC and AUPRC
        try:
            if len(np.unique(labels)) > 1:
                auc = roc_auc_score(labels, pred_probs)
                avg_precision = average_precision_score(labels, pred_probs)
            else:
                auc = 0.5
                avg_precision = 0.0
        except:
            auc = 0.5
            avg_precision = 0.0
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denominator > 0:
            mcc = (tp * tn - fp * fn) / mcc_denominator
        else:
            mcc = 0.0
        
        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2
        
        # Cost-sensitive metrics for blockchain
        # Assume false negatives (missed fraud) are more costly
        fn_cost = 10.0  # Missing fraud is 10x more costly
        fp_cost = 1.0   # False alarm cost
        
        total_cost = fn * fn_cost + fp * fp_cost
        normalized_cost = total_cost / (total * fn_cost)  # Normalize by worst case
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'auc': float(auc),
            'avg_precision': float(avg_precision),
            'specificity': float(specificity),
            'npv': float(npv),
            'mcc': float(mcc),
            'balanced_accuracy': float(balanced_acc),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'total_cost': float(total_cost),
            'normalized_cost': float(normalized_cost)
        }
        
        # Add sample weight statistics if provided
        if sample_weights is not None:
            weights = sample_weights.cpu().numpy()
            metrics['avg_sample_weight'] = float(np.mean(weights))
            metrics['weight_variance'] = float(np.var(weights))
        
        return metrics


class EfficiencyMetrics:
    """Calculates computational efficiency metrics for large-scale processing."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_properties = [
                torch.cuda.get_device_properties(i) 
                for i in range(torch.cuda.device_count())
            ]
    
    def compute(
        self,
        timing_info: Dict[str, float],
        batch_size: int
    ) -> Dict[str, float]:
        """Compute comprehensive efficiency metrics."""
        metrics = {}
        
        # Time metrics
        total_time = timing_info.get('total_time', 0)
        metrics['processing_time'] = total_time
        metrics['processing_time_ms'] = total_time * 1000
        
        # Throughput metrics
        if total_time > 0:
            metrics['samples_per_second'] = batch_size / total_time
            metrics['ms_per_sample'] = (total_time * 1000) / batch_size
        else:
            metrics['samples_per_second'] = 0
            metrics['ms_per_sample'] = 0
        
        # Breakdown of time components
        if 'data_loading_time' in timing_info:
            metrics['data_loading_percentage'] = (
                timing_info['data_loading_time'] / total_time * 100 
                if total_time > 0 else 0
            )
        
        if 'forward_time' in timing_info:
            metrics['forward_percentage'] = (
                timing_info['forward_time'] / total_time * 100 
                if total_time > 0 else 0
            )
        
        if 'backward_time' in timing_info:
            metrics['backward_percentage'] = (
                timing_info['backward_time'] / total_time * 100 
                if total_time > 0 else 0
            )
        
        # Memory metrics
        memory_info = psutil.virtual_memory()
        metrics['memory_usage_percent'] = memory_info.percent
        metrics['memory_used_gb'] = memory_info.used / (1024**3)
        metrics['memory_available_gb'] = memory_info.available / (1024**3)
        
        # CPU metrics
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        metrics['cpu_count'] = psutil.cpu_count()
        
        # GPU metrics
        if self.gpu_available:
            try:
                gpu_memory_used = []
                gpu_memory_total = []
                gpu_utilization = []
                
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    mem_total = self.gpu_properties[i].total_memory / (1024**3)
                    
                    gpu_memory_used.append(mem_reserved)
                    gpu_memory_total.append(mem_total)
                    
                    # Try to get GPU utilization using nvidia-ml-py
                    try:
                        import nvidia_ml_py as nvml
                        nvml.nvmlInit()
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_utilization.append(util.gpu)
                    except:
                        gpu_utilization.append(0)
                
                metrics['gpu_memory_used_gb'] = sum(gpu_memory_used)
                metrics['gpu_memory_total_gb'] = sum(gpu_memory_total)
                metrics['gpu_memory_percent'] = (
                    metrics['gpu_memory_used_gb'] / metrics['gpu_memory_total_gb'] * 100
                    if metrics['gpu_memory_total_gb'] > 0 else 0
                )
                metrics['gpu_utilization'] = np.mean(gpu_utilization) if gpu_utilization else 0
                
                # Per-GPU metrics
                for i in range(len(gpu_memory_used)):
                    metrics[f'gpu{i}_memory_gb'] = gpu_memory_used[i]
                    metrics[f'gpu{i}_utilization'] = gpu_utilization[i] if i < len(gpu_utilization) else 0
                    
            except Exception as e:
                print(f"Error getting GPU metrics: {e}")
                metrics['gpu_memory_used_gb'] = 0
                metrics['gpu_memory_percent'] = 0
                metrics['gpu_utilization'] = 0
        
        # Communication metrics for distributed training
        if 'communication_time' in timing_info:
            comm_time = timing_info['communication_time']
            metrics['communication_time_ms'] = comm_time * 1000
            metrics['communication_overhead'] = comm_time / total_time if total_time > 0 else 0
            
            if 'communication_volume' in timing_info:
                comm_volume_mb = timing_info['communication_volume'] / (1024**2)
                metrics['communication_volume_mb'] = comm_volume_mb
                metrics['communication_bandwidth_mbps'] = (
                    comm_volume_mb / comm_time if comm_time > 0 else 0
                )
        
        # I/O metrics
        if 'io_wait_time' in timing_info:
            metrics['io_wait_percentage'] = (
                timing_info['io_wait_time'] / total_time * 100 
                if total_time > 0 else 0
            )
        
        # Cache efficiency
        if 'cache_hits' in timing_info and 'cache_misses' in timing_info:
            total_accesses = timing_info['cache_hits'] + timing_info['cache_misses']
            metrics['cache_hit_rate'] = (
                timing_info['cache_hits'] / total_accesses 
                if total_accesses > 0 else 0
            )
        
        return metrics


# Transform functions
class AddGaussianNoise:
    """Add Gaussian noise to features for data augmentation."""
    
    def __init__(self, std: float = 0.01, feature_wise: bool = True):
        self.std = std
        self.feature_wise = feature_wise
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'x') and data.x is not None:
            if self.feature_wise:
                # Different noise level for each feature
                noise = torch.randn_like(data.x) * self.std
            else:
                # Same noise level for all features
                noise = torch.randn(data.x.shape[0], 1) * self.std
                noise = noise.expand_as(data.x)
            
            data.x = data.x + noise
        
        return data


class RandomEdgeDrop:
    """Randomly drop edges for robustness training."""
    
    def __init__(self, p: float = 0.1, directed: bool = True):
        self.p = p
        self.directed = directed
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            num_edges = data.edge_index.size(1)
            
            if num_edges > 0:
                # Generate drop mask
                keep_mask = torch.rand(num_edges) > self.p
                
                # Apply mask
                data.edge_index = data.edge_index[:, keep_mask]
                
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[keep_mask]
                
                # Update edge count
                if hasattr(data, 'num_edges'):
                    data.num_edges = data.edge_index.size(1)
        
        return data


class RandomNodeDrop:
    """Randomly drop nodes for robustness training."""
    
    def __init__(self, p: float = 0.05):
        self.p = p
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.size(0)
            
            if num_nodes > 1:  # Keep at least one node
                # Generate keep mask
                keep_mask = torch.rand(num_nodes)
                keep_mask = keep_mask > self.p
                
                # Ensure at least one node is kept
                if not keep_mask.any():
                    keep_mask[0] = True
                
                # Get node indices to keep
                keep_indices = torch.where(keep_mask)[0]
                
                # Update node features
                data.x = data.x[keep_mask]
                
                # Update edges
                if hasattr(data, 'edge_index') and data.edge_index is not None:
                    # Create node index mapping
                    node_map = torch.full((num_nodes,), -1, dtype=torch.long)
                    node_map[keep_indices] = torch.arange(len(keep_indices))
                    
                    # Filter edges
                    edge_mask = (node_map[data.edge_index[0]] >= 0) & (node_map[data.edge_index[1]] >= 0)
                    data.edge_index = node_map[data.edge_index[:, edge_mask]]
                    
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        data.edge_attr = data.edge_attr[edge_mask]
                
                # Update node count
                if hasattr(data, 'num_nodes'):
                    data.num_nodes = len(keep_indices)
        
        return data


class AugmentSequences:
    """Augment transaction sequences for better generalization."""
    
    def __init__(self, noise_std: float = 0.01, time_shift_std: float = 0.1):
        self.noise_std = noise_std
        self.time_shift_std = time_shift_std
    
    def __call__(self, data: Data) -> Data:
        # Augment in_sequences
        if hasattr(data, 'in_sequences') and data.in_sequences is not None:
            # Add noise
            noise = torch.randn_like(data.in_sequences) * self.noise_std
            data.in_sequences = data.in_sequences + noise
            
            # Time shift augmentation
            if data.in_sequences.size(0) > 1:
                shifts = torch.randn(data.in_sequences.size(0)) * self.time_shift_std
                for i in range(data.in_sequences.size(0)):
                    if i + int(shifts[i]) >= 0 and i + int(shifts[i]) < data.in_sequences.size(0):
                        data.in_sequences[i] = data.in_sequences[i + int(shifts[i])]
        
        # Augment out_sequences
        if hasattr(data, 'out_sequences') and data.out_sequences is not None:
            noise = torch.randn_like(data.out_sequences) * self.noise_std
            data.out_sequences = data.out_sequences + noise
        
        return data


class NormalizeFeatures:
    """Normalize node and edge features."""
    
    def __init__(self, norm: str = 'standard', dim: int = -1):
        self.norm = norm
        self.dim = dim
    
    def __call__(self, data: Data) -> Data:
        # Normalize node features
        if hasattr(data, 'x') and data.x is not None:
            if self.norm == 'standard':
                mean = data.x.mean(dim=self.dim, keepdim=True)
                std = data.x.std(dim=self.dim, keepdim=True) + 1e-6
                data.x = (data.x - mean) / std
            elif self.norm == 'minmax':
                min_val = data.x.min(dim=self.dim, keepdim=True)[0]
                max_val = data.x.max(dim=self.dim, keepdim=True)[0]
                data.x = (data.x - min_val) / (max_val - min_val + 1e-6)
            elif self.norm == 'l2':
                data.x = F.normalize(data.x, p=2, dim=self.dim)
        
        # Normalize edge features
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if self.norm == 'standard':
                mean = data.edge_attr.mean(dim=self.dim, keepdim=True)
                std = data.edge_attr.std(dim=self.dim, keepdim=True) + 1e-6
                data.edge_attr = (data.edge_attr - mean) / std
            elif self.norm == 'minmax':
                min_val = data.edge_attr.min(dim=self.dim, keepdim=True)[0]
                max_val = data.edge_attr.max(dim=self.dim, keepdim=True)[0]
                data.edge_attr = (data.edge_attr - min_val) / (max_val - min_val + 1e-6)
            elif self.norm == 'l2':
                data.edge_attr = F.normalize(data.edge_attr, p=2, dim=self.dim)
        
        return data


def compose_transforms(transforms: List[Callable]) -> Callable:
    """Compose multiple transforms into a single transform."""
    def composed(data: Data) -> Data:
        for transform in transforms:
            data = transform(data)
        return data
    return composed


# Additional utility classes
class DistributedBatchSampler:
    """Custom batch sampler for distributed training with balanced sampling."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        # Calculate samples per rank
        self.total_size = len(dataset)
        self.num_samples = self.total_size // world_size
        if not drop_last and self.total_size % world_size != 0:
            self.num_samples += 1
        
        self.total_batches = self.num_samples // batch_size
        if not drop_last and self.num_samples % batch_size != 0:
            self.total_batches += 1
    
    def __iter__(self):
        # Set seed for reproducibility
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))
        
        # Subset for this rank
        rank_indices = indices[self.rank::self.world_size]
        
        # Create batches
        batches = []
        for i in range(0, len(rank_indices), self.batch_size):
            batch = rank_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        return iter(batches)
    
    def __len__(self):
        return self.total_batches
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.seed = epoch


class StreamingFeatureBuffer:
    """Maintains feature statistics for streaming data with efficient memory usage."""
    
    def __init__(self, window_size: int, feature_dim: int = 64):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.node_features = defaultdict(lambda: deque(maxlen=window_size))
        self.edge_features = defaultdict(lambda: deque(maxlen=window_size))
        self.global_stats = WindowStatistics()
        self.last_cleanup = time.time()
    
    def update(self, transaction: Dict[str, Any]):
        """Update buffer with new transaction."""
        source = transaction['source']
        dest = transaction['destination']
        timestamp = transaction.get('timestamp', time.time())
        
        # Extract and store features
        source_features = self._extract_node_features(source, transaction, 'source')
        dest_features = self._extract_node_features(dest, transaction, 'destination')
        edge_features = self._extract_edge_features(transaction)
        
        # Update buffers
        self.node_features[source].append((timestamp, source_features))
        self.node_features[dest].append((timestamp, dest_features))
        self.edge_features[(source, dest)].append((timestamp, edge_features))
        
        # Update global statistics
        self.global_stats.update(transaction)
    
    def get_node_features(self, node_id: str) -> torch.Tensor:
        """Get aggregated features for a node with temporal weighting."""
        if node_id not in self.node_features or not self.node_features[node_id]:
            return torch.zeros(self.feature_dim)
        
        features_with_time = list(self.node_features[node_id])
        current_time = time.time()
        
        # Temporal weighting - recent features have higher weight
        weighted_features = []
        weights = []
        
        for timestamp, features in features_with_time:
            time_diff = current_time - timestamp
            weight = np.exp(-time_diff / 3600)  # 1 hour decay
            weighted_features.append(features * weight)
            weights.append(weight)
        
        if not weights:
            return torch.zeros(self.feature_dim)
        
        # Weighted average
        weights = torch.tensor(weights)
        weights = weights / weights.sum()
        
        aggregated = sum(w * f for w, f in zip(weights, weighted_features))
        
        return aggregated
    
    def get_window_statistics(self) -> torch.Tensor:
        """Get current window statistics."""
        return self.global_stats.get_summary()
    
    def _extract_node_features(self, node: str, transaction: Dict[str, Any], role: str) -> torch.Tensor:
        """Extract features for a node based on its role in the transaction."""
        features = []
        
        # Role-based features
        features.append(1.0 if role == 'source' else 0.0)
        features.append(1.0 if role == 'destination' else 0.0)
        
        # Transaction amount (log scale)
        amount = transaction.get('amount', 0.0)
        features.append(np.log1p(amount))
        
        # Time features
        timestamp = transaction.get('timestamp', time.time())
        hour = (timestamp % 86400) / 3600
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        
        # Day of week
        day = (timestamp // 86400) % 7
        day_features = [0.0] * 7
        day_features[int(day)] = 1.0
        features.extend(day_features)
        
        # Transaction frequency (from window stats)
        node_frequency = len(self.node_features.get(node, []))
        features.append(np.log1p(node_frequency))
        
        # Gas/fee features if available
        gas_price = transaction.get('gas_price', 0.0)
        features.append(np.log1p(gas_price))
        
        # Pad to feature dimension
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.feature_dim], dtype=torch.float32)
    
    def _extract_edge_features(self, transaction: Dict[str, Any]) -> torch.Tensor:
        """Extract edge features from transaction."""
        features = []
        
        # Amount features
        amount = transaction.get('amount', 0.0)
        features.append(np.log1p(amount))
        features.append(amount > 1000)  # Large transaction indicator
        
        # Time features
        timestamp = transaction.get('timestamp', time.time())
        features.append(timestamp % 3600 / 3600)  # Time within hour
        
        # Transaction type
        tx_type = transaction.get('type', 'transfer')
        type_map = {'transfer': 0, 'contract': 1, 'mining': 2, 'exchange': 3}
        type_val = type_map.get(tx_type, 4)
        type_features = [0.0] * 5
        type_features[type_val] = 1.0
        features.extend(type_features)
        
        # Confirmation time if available
        conf_time = transaction.get('confirmation_time', 60.0)
        features.append(np.log1p(conf_time))
        
        # Pad to standard edge feature dimension
        edge_dim = 8
        while len(features) < edge_dim:
            features.append(0.0)
        
        return torch.tensor(features[:edge_dim], dtype=torch.float32)
    
    def cleanup_old_data(self):
        """Remove old data to free memory."""
        current_time = time.time()
        
        # Only cleanup every 5 minutes
        if current_time - self.last_cleanup < 300:
            return
        
        # Remove entries older than window
        cutoff_time = current_time - self.window_size * 60  # Assuming window_size in minutes
        
        # Cleanup node features
        for node_id in list(self.node_features.keys()):
            features = self.node_features[node_id]
            # Remove old entries
            while features and features[0][0] < cutoff_time:
                features.popleft()
            # Remove empty entries
            if not features:
                del self.node_features[node_id]
        
        # Cleanup edge features
        for edge in list(self.edge_features.keys()):
            features = self.edge_features[edge]
            while features and features[0][0] < cutoff_time:
                features.popleft()
            if not features:
                del self.edge_features[edge]
        
        self.last_cleanup = current_time


class WindowStatistics:
    """Tracks statistics over sliding windows with incremental updates."""
    
    def __init__(self):
        self.window_id = 0
        self.transaction_count = 0
        self.amount_sum = 0.0
        self.amount_squared_sum = 0.0
        self.unique_addresses = set()
        self.anomaly_count = 0
        self.start_time = time.time()
        self.chain_counts = defaultdict(int)
    
    def update(self, transaction: Dict[str, Any]):
        """Update statistics with new transaction."""
        self.transaction_count += 1
        
        # Amount statistics
        amount = transaction.get('amount', 0.0)
        self.amount_sum += amount
        self.amount_squared_sum += amount ** 2
        
        # Address statistics
        self.unique_addresses.add(transaction['source'])
        self.unique_addresses.add(transaction['destination'])
        
        # Anomaly tracking
        if transaction.get('is_anomalous', False):
            self.anomaly_count += 1
        
        # Chain statistics
        chain = transaction.get('chain_id', 'unknown')
        self.chain_counts[chain] += 1
    
    def get_summary(self) -> torch.Tensor:
        """Get summary statistics as tensor."""
        if self.transaction_count == 0:
            return torch.zeros(12)
        
        # Calculate statistics
        mean_amount = self.amount_sum / self.transaction_count
        var_amount = (self.amount_squared_sum / self.transaction_count) - mean_amount ** 2
        std_amount = np.sqrt(max(0, var_amount))
        
        # Time-based statistics
        current_time = time.time()
        window_duration = current_time - self.start_time
        tx_rate = self.transaction_count / max(window_duration, 1.0)
        
        # Diversity metrics
        address_diversity = len(self.unique_addresses) / (2 * self.transaction_count)
        chain_diversity = len(self.chain_counts) / max(len(self.chain_counts), 1)
        
        summary = torch.tensor([
            self.transaction_count,
            mean_amount,
            std_amount,
            len(self.unique_addresses),
            self.anomaly_count,
            self.anomaly_count / self.transaction_count,
            tx_rate,
            address_diversity,
            chain_diversity,
            window_duration,
            self.window_id,
            current_time
        ], dtype=torch.float32)
        
        return summary
    
    def reset(self):
        """Reset statistics for new window."""
        self.window_id += 1
        self.transaction_count = 0
        self.amount_sum = 0.0
        self.amount_squared_sum = 0.0
        self.unique_addresses.clear()
        self.anomaly_count = 0
        self.start_time = time.time()
        self.chain_counts.clear()


class IncrementalGraphBuilder:
    """Builds graphs incrementally from transaction streams with efficient updates."""
    
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.node_id_map = {}
        self.next_node_id = 0
        self.edge_count = defaultdict(int)
    
    def build_from_transactions(
        self,
        transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build graph representation from transactions."""
        edge_list = []
        edge_weights = []
        node_set = set()
        
        for tx_data in transactions:
            tx = tx_data['data'] if 'data' in tx_data else tx_data
            source = tx.get('source')
            dest = tx.get('destination')
            
            if source is None or dest is None:
                continue
            
            # Add nodes
            node_set.add(source)
            node_set.add(dest)
            
            # Add edge with weight
            edge_list.append((source, dest))
            weight = tx.get('amount', 1.0)
            edge_weights.append(weight)
            
            # Track edge frequency
            edge_key = (source, dest)
            self.edge_count[edge_key] += 1
        
        # Create consistent node mapping
        node_list = sorted(list(node_set))
        
        # Update global node mapping
        for node in node_list:
            if node not in self.node_id_map:
                self.node_id_map[node] = self.next_node_id
                self.next_node_id += 1
        
        # Use local mapping for this graph
        local_node_map = {node: idx for idx, node in enumerate(node_list)}
        
        # Convert edges to indices
        edge_index = []
        edge_attr = []
        
        for (source, dest), weight in zip(edge_list, edge_weights):
            if source in local_node_map and dest in local_node_map:
                edge_index.append([local_node_map[source], local_node_map[dest]])
                
                # Edge attributes: [weight, frequency, normalized_weight]
                freq = self.edge_count[(source, dest)]
                norm_weight = weight / max(edge_weights) if edge_weights else 1.0
                edge_attr.append([weight, freq, norm_weight])
                
                if not self.directed:
                    edge_index.append([local_node_map[dest], local_node_map[source]])
                    edge_attr.append([weight, freq, norm_weight])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        
        return {
            'nodes': node_list,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'node_to_idx': local_node_map,
            'global_node_ids': {node: self.node_id_map[node] for node in node_list}
        }


class LRUCache:
    """Thread-safe LRU cache implementation with size tracking."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update and move to end
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new item
                self.cache[key] = value
                if len(self.cache) > self.capacity:
                    # Remove least recently used
                    self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'capacity': self.capacity
        }
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class StreamAnomalyDetector:
    """Detects anomalies in streaming data with adaptive thresholds."""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.score_history = deque(maxlen=window_size)
        self.threshold = 0.7
        self.threshold_history = deque(maxlen=100)
        
    def update_threshold(self, scores: Union[List[float], np.ndarray]):
        """Update anomaly threshold based on recent scores."""
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        self.score_history.extend(scores)
        
        if len(self.score_history) >= 10:
            # Use robust statistics
            scores_array = np.array(self.score_history)
            
            # Remove outliers using IQR
            q1, q3 = np.percentile(scores_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter scores
            filtered_scores = scores_array[
                (scores_array >= lower_bound) & (scores_array <= upper_bound)
            ]
            
            if len(filtered_scores) > 0:
                # Calculate threshold based on contamination rate
                self.threshold = np.percentile(
                    filtered_scores, 
                    (1 - self.contamination) * 100
                )
                self.threshold = np.clip(self.threshold, 0.3, 0.95)
                self.threshold_history.append(self.threshold)
    
    def is_anomalous(self, score: float) -> bool:
        """Check if score indicates anomaly."""
        return score > self.threshold
    
    def get_threshold_trend(self) -> float:
        """Get trend of threshold changes."""
        if len(self.threshold_history) < 2:
            return 0.0
        
        # Linear regression on threshold history
        x = np.arange(len(self.threshold_history))
        y = np.array(self.threshold_history)
        
        # Calculate trend
        trend = np.polyfit(x, y, 1)[0]
        return trend


class DataPrefetcher:
    """Asynchronously prefetches data for improved performance."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        self.cache = LRUCache(capacity=1000)
    
    def prefetch(self, keys: List[str], load_fn: Callable):
        """Prefetch multiple items asynchronously."""
        for key in keys:
            if key not in self.futures and self.cache.get(key) is None:
                future = self.executor.submit(self._load_with_cache, key, load_fn)
                self.futures[key] = future
    
    def _load_with_cache(self, key: str, load_fn: Callable) -> Any:
        """Load item and cache it."""
        data = load_fn(key)
        self.cache.put(key, data)
        return data
    
    def get(self, key: str, timeout: float = 5.0) -> Optional[Any]:
        """Get prefetched item with timeout."""
        # Check cache first
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        # Check futures
        if key in self.futures:
            try:
                future = self.futures.pop(key)
                return future.result(timeout=timeout)
            except Exception as e:
                print(f"Error getting prefetched item {key}: {e}")
                return None
        
        return None
    
    def shutdown(self):
        """Shutdown prefetcher and cleanup."""
        # Cancel pending futures
        for future in self.futures.values():
            future.cancel()
        
        self.executor.shutdown(wait=True)
        self.cache.clear()


class PrefetchBuffer:
    """Buffer for prefetched data with priority management."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.buffer = {}
        self.priorities = {}
        self.access_counts = defaultdict(int)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
    
    def prefetch(self, idx: int, load_fn: Callable, priority: float = 1.0):
        """Submit prefetch task."""
        with self.lock:
            if idx not in self.buffer and idx not in self.priorities:
                self.priorities[idx] = priority
                future = self.prefetch_executor.submit(load_fn, idx)
                self.buffer[idx] = future
    
    def get(self, idx: int) -> Optional[Any]:
        """Get prefetched data if available."""
        with self.lock:
            if idx in self.buffer:
                future = self.buffer.pop(idx)
                self.priorities.pop(idx, None)
                
                try:
                    data = future.result(timeout=0.1)
                    self.access_counts[idx] += 1
                    return data
                except:
                    return None
        
        return None
    
    def contains(self, idx: int) -> bool:
        """Check if index is in buffer."""
        with self.lock:
            return idx in self.buffer
    
    def cleanup(self):
        """Remove low-priority items if buffer is full."""
        with self.lock:
            if len(self.buffer) >= self.window_size:
                # Remove lowest priority items
                sorted_items = sorted(self.priorities.items(), key=lambda x: x[1])
                
                for idx, _ in sorted_items[:len(self.buffer) - self.window_size + 1]:
                    if idx in self.buffer:
                        future = self.buffer.pop(idx)
                        future.cancel()
                        self.priorities.pop(idx, None)


class LoadingStatistics:
    """Track data loading statistics for optimization."""
    
    def __init__(self):
        self.load_times = deque(maxlen=1000)
        self.batch_sizes = deque(maxlen=1000)
        self.start_time = time.time()
        self.total_samples = 0
    
    def update(self, batch_size: int, load_time: float):
        """Update loading statistics."""
        self.load_times.append(load_time)
        self.batch_sizes.append(batch_size)
        self.total_samples += batch_size
    
    def get_stats(self) -> Dict[str, float]:
        """Get loading statistics."""
        if not self.load_times:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'avg_load_time': np.mean(self.load_times),
            'std_load_time': np.std(self.load_times),
            'avg_batch_size': np.mean(self.batch_sizes),
            'total_samples': self.total_samples,
            'samples_per_second': self.total_samples / elapsed_time,
            'elapsed_time': elapsed_time
        }


class GPUCache:
    """GPU memory cache for frequently accessed data."""
    
    def __init__(self, max_size_gb: float):
        if not torch.cuda.is_available():
            self.enabled = False
            return
        
        self.enabled = True
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size = 0
        self.cache = OrderedDict()
        self.device = torch.cuda.current_device()
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from GPU cache."""
        if not self.enabled:
            return None
        
        with self.lock:
            if key in self.cache:
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return self.cache[key]
        
        return None
    
    def put(self, key: str, data: Any):
        """Put item in GPU cache."""
        if not self.enabled:
            return
        
        with self.lock:
            # Move data to GPU if needed
            if hasattr(data, 'to'):
                gpu_data = data.to(self.device)
                size = self._estimate_size(gpu_data)
                
                # Evict if necessary
                while self.current_size + size > self.max_size_bytes and self.cache:
                    self._evict_oldest()
                
                # Add to cache
                if self.current_size + size <= self.max_size_bytes:
                    self.cache[key] = gpu_data
                    self.current_size += size
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate GPU memory size of data."""
        if isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        elif hasattr(data, 'x'):  # PyG Data object
            size = 0
            for attr in ['x', 'edge_index', 'edge_attr']:
                if hasattr(data, attr):
                    tensor = getattr(data, attr)
                    if tensor is not None:
                        size += tensor.element_size() * tensor.nelement()
            return size
        return 0
    
    def _evict_oldest(self):
        """Evict oldest item from cache."""
        if self.cache:
            key, data = self.cache.popitem(last=False)
            self.current_size -= self._estimate_size(data)


class MemoryMappedAccessor(DataAccessor):
    """Memory-mapped data access for efficient large file handling."""
    
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.data_files = self._index_data_files()
        self.mmaps = {}
        
    def _index_data_files(self) -> Dict[str, Path]:
        """Index available data files."""
        files = {}
        
        # Check for different file formats
        for pattern in ['*.bin', '*.dat', '*.h5']:
            for file_path in self.data_path.glob(pattern):
                files[file_path.stem] = file_path
        
        return files
    
    def load_sample(self, idx: int) -> Data:
        """Load sample using memory mapping."""
        # Try binary format first
        binary_file = self.data_path / f'sample_{idx}.bin'
        if binary_file.exists():
            return self._load_binary_sample(binary_file)
        
        # Try HDF5 format
        h5_file = self.data_path / 'data.h5'
        if h5_file.exists():
            return self._load_h5_sample(h5_file, idx)
        
        # Fallback to torch format
        pt_file = self.data_path / f'sample_{idx}.pt'
        if pt_file.exists():
            return torch.load(pt_file, map_location='cpu')
        
        raise FileNotFoundError(f"No data file found for sample {idx}")
    
    def _load_binary_sample(self, file_path: Path) -> Data:
        """Load sample from binary file with memory mapping."""
        if file_path not in self.mmaps:
            self.mmaps[file_path] = np.memmap(
                file_path, dtype='float32', mode='r'
            )
        
        mmap_data = self.mmaps[file_path]
        
        # Parse binary format (implementation-specific)
        # This is a placeholder - actual implementation depends on data format
        data = Data()
        # ... parse mmap_data into PyG Data object
        
        return data
    
    def _load_h5_sample(self, file_path: Path, idx: int) -> Data:
        """Load sample from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Load data components
            x = torch.tensor(f['features'][idx])
            edge_index = torch.tensor(f['edge_index'][idx])
            
            data = Data(x=x, edge_index=edge_index)
            
            # Load optional components
            if 'edge_attr' in f:
                data.edge_attr = torch.tensor(f['edge_attr'][idx])
            if 'y' in f:
                data.y = torch.tensor(f['y'][idx])
            
            return data
    
    def load_transaction(self, idx: int) -> Dict[str, Any]:
        """Load transaction data."""
        # Implementation depends on data format
        tx_file = self.data_path / f'tx_{idx}.json'
        if tx_file.exists():
            with open(tx_file, 'r') as f:
                return json.load(f)
        
        # Fallback to sample data
        sample = self.load_sample(idx)
        return {
            'source': 'node_0',
            'destination': 'node_1',
            'amount': 1.0,
            'timestamp': time.time()
        }


class DirectAccessor(DataAccessor):
    """Direct file access for standard data loading."""
    
    def load_sample(self, idx: int) -> Data:
        """Load sample directly from file."""
        # Try different file formats
        for ext in ['.pt', '.pth', '.pkl']:
            sample_file = self.data_path / f'sample_{idx}{ext}'
            if sample_file.exists():
                if ext in ['.pt', '.pth']:
                    return torch.load(sample_file, map_location='cpu')
                else:
                    with open(sample_file, 'rb') as f:
                        return pickle.load(f)
        
        raise FileNotFoundError(f"Sample {idx} not found")
    
    def load_transaction(self, idx: int) -> Dict[str, Any]:
        """Load transaction data."""
        tx_file = self.data_path / f'transaction_{idx}.json'
        if tx_file.exists():
            with open(tx_file, 'r') as f:
                return json.load(f)
        
        # Generate synthetic transaction for testing
        return {
            'source': f'addr_{random.randint(0, 1000)}',
            'destination': f'addr_{random.randint(0, 1000)}',
            'amount': random.uniform(0.1, 1000),
            'timestamp': time.time() - random.uniform(0, 86400),
            'chain_id': random.choice(['ethereum', 'bitcoin'])
        }


# Utility functions
def create_data_loaders(
    config: Dict[str, Any],
    num_gpus: int = 4,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create optimized data loaders for multi-GPU training."""
    
    # Initialize data loader
    loader = MemoryEfficientDataLoader(
        data_path=Path(config['data_path']),
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 32),
        cache_size_gb=config.get('cache_size_gb', 100),
        prefetch_batches=config.get('prefetch_batches', 10),
        streaming_mode=config.get('streaming_mode', False)
    )
    
    # Load datasets
    train_dataset = loader.load_dataset('train')
    val_dataset = loader.load_dataset('val')
    test_dataset = loader.load_dataset('test')
    
    # Create distributed loaders
    train_loader = loader.create_distributed_loader(
        train_dataset, rank=rank, world_size=num_gpus, shuffle=True
    )
    
    val_loader = loader.create_distributed_loader(
        val_dataset, rank=rank, world_size=num_gpus, shuffle=False
    )
    
    test_loader = loader.create_distributed_loader(
        test_dataset, rank=rank, world_size=num_gpus, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def generate_sample_network(
    num_nodes: int = 1000,
    topology: str = 'blockchain'
) -> nx.Graph:
    """Generate sample network for testing."""
    return NetworkTopologyGenerator.generate(
        topology_type=topology,
        num_nodes=num_nodes,
        num_miners=int(num_nodes * 0.05),
        num_full_nodes=int(num_nodes * 0.15),
        num_exchanges=int(num_nodes * 0.02)
    )


def calculate_network_statistics(G: nx.Graph) -> Dict[str, float]:
    """Calculate comprehensive network statistics."""
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes(),
        'max_degree': max(dict(G.degree()).values()),
        'clustering_coefficient': nx.average_clustering(G),
        'diameter': nx.diameter(G) if nx.is_connected(G) else -1,
        'num_components': nx.number_connected_components(G)
    }
    
    # Node type distribution if available
    if 'node_type' in next(iter(G.nodes(data=True)))[1]:
        node_types = nx.get_node_attributes(G, 'node_type')
        type_counts = defaultdict(int)
        for node_type in node_types.values():
            type_counts[node_type] += 1
        
        for node_type, count in type_counts.items():
            stats[f'num_{node_type}_nodes'] = count
    
    return stats


def setup_distributed_metrics() -> 'MetricsCalculator':
    """Setup metrics calculator for distributed training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return MetricsCalculator(device=device)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Path
):
    """Save training checkpoint with metrics."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint['metrics'],
        'timestamp': checkpoint.get('timestamp', 0)
    }
