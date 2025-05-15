#!/usr/bin/env python
"""
Debug Data Structure Script

This script provides comprehensive debugging information about the wind forecasting dataset,
helping to understand:
1. Which parameter takes precedence between explicit values and context_length_factor
2. Detailed information about dataset structure
3. Number of valid windows per group
4. Number of training steps
5. Other relevant metrics

Usage:
    python -m wind_forecasting.run_scripts.debug_data_structure --config PATH_TO_CONFIG [--model MODEL_TYPE]
"""

import argparse
import logging
import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from collections import defaultdict
import re
import json
from typing import Dict, List, Tuple, Any, Optional

# Internal imports
from wind_forecasting.preprocessing.data_module import DataModule
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler, SequentialSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDebugger:
    """Class to debug data structure and provide insights about the dataset."""
    
    def __init__(self, config_path: str, model_type: str = "tactis"):
        """
        Initialize the debugger with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            model_type: Type of model (tactis, informer, etc.)
        """
        self.config_path = config_path
        self.model_type = model_type
        self.config = self._load_config()
        self.data_module = None
        self.samplers = {}
        self.window_counts = {}
        self.continuity_group_info = {}
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    def initialize_data_module(self) -> None:
        """Initialize the data module with configuration parameters."""
        logger.info("Initializing data module")
        
        # Extract dataset config
        dataset_config = self.config["dataset"]
        
        # Determine normalization based on model type
        use_normalization = False if self.model_type == "tactis" else dataset_config.get("normalize", True)
        
        # Initialize data module
        self.data_module = DataModule(
            data_path=dataset_config["data_path"],
            n_splits=dataset_config["n_splits"],
            continuity_groups=None,
            train_split=(1.0 - dataset_config["val_split"] - dataset_config["test_split"]),
            val_split=dataset_config["val_split"],
            test_split=dataset_config["test_split"],
            prediction_length=dataset_config["prediction_length"],
            context_length=dataset_config["context_length"],
            target_prefixes=["ws_horz", "ws_vert"],
            feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
            freq=dataset_config["resample_freq"],
            target_suffixes=dataset_config["target_turbine_ids"],
            per_turbine_target=dataset_config["per_turbine_target"],
            as_lazyframe=False,
            normalized=use_normalization,
            normalization_consts_path=dataset_config["normalization_consts_path"],
            batch_size=dataset_config["batch_size"],
            workers=4,
            pin_memory=True,
            persistent_workers=True,
            verbose=True
        )
        
        # Check if data is already prepared
        if not os.path.exists(self.data_module.train_ready_data_path):
            logger.info("Generating datasets")
            self.data_module.generate_datasets()
            reload = True
        else:
            reload = False
            
        # Generate splits
        logger.info("Generating splits")
        self.data_module.generate_splits(save=True, reload=reload, splits=["train", "val", "test"])
        
    def analyze_context_length_precedence(self) -> Dict:
        """
        Analyze which context length parameter takes precedence.
        
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing context length precedence")
        
        # Extract relevant parameters
        explicit_context_length = self.config["dataset"].get("context_length")
        explicit_prediction_length = self.config["dataset"].get("prediction_length")
        context_length_factor = self.config["dataset"].get("context_length_factor")
        freq = self.config["dataset"].get("resample_freq")
        
        # Calculate time steps
        context_steps = int(pd.Timedelta(explicit_context_length, unit="s") / pd.Timedelta(freq))
        prediction_steps = int(pd.Timedelta(explicit_prediction_length, unit="s") / pd.Timedelta(freq))
        factor_based_context_steps = int(context_length_factor * prediction_steps) if context_length_factor else None
        
        # Determine which would take precedence
        if context_length_factor:
            precedence = "context_length_factor"
            effective_context_steps = factor_based_context_steps
        else:
            precedence = "explicit context_length"
            effective_context_steps = context_steps
            
        # Check what's actually used in the data module
        actual_context_steps = self.data_module.context_length
        actual_prediction_steps = self.data_module.prediction_length
        
        # Determine if there's a mismatch
        mismatch = actual_context_steps != effective_context_steps
        
        return {
            "explicit_context_length_seconds": explicit_context_length,
            "explicit_prediction_length_seconds": explicit_prediction_length,
            "context_length_factor": context_length_factor,
            "frequency": freq,
            "context_steps_from_explicit": context_steps,
            "prediction_steps": prediction_steps,
            "context_steps_from_factor": factor_based_context_steps,
            "precedence": precedence,
            "expected_context_steps": effective_context_steps,
            "actual_context_steps": actual_context_steps,
            "actual_prediction_steps": actual_prediction_steps,
            "mismatch": mismatch
        }
    
    def analyze_dataset_structure(self) -> Dict:
        """
        Analyze the structure of the dataset.
        
        Returns:
            Dictionary with dataset structure information
        """
        logger.info("Analyzing dataset structure")
        
        result = {
            "continuity_groups": [],
            "total_samples": 0,
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "target_variables": self.data_module.target_cols,
            "feature_variables": self.data_module.feat_dynamic_real_cols,
            "per_turbine_target": self.data_module.per_turbine_target,
            "turbine_ids": self.data_module.target_suffixes
        }
        
        # Calculate dataset sizes for each continuity group
        for cg_idx, cg in enumerate(self.data_module.continuity_groups):
            # Convert floating-point continuity group to integer for pattern matching
            cg_int = int(cg)
            
            # Find datasets for this continuity group
            cg_train_datasets = [ds for ds in self.data_module.train_dataset
                               if f"SPLIT{cg_int}" in ds["item_id"]]
            
            if not cg_train_datasets:
                # Try alternative pattern for per_turbine_target=True
                cg_train_datasets = [ds for ds in self.data_module.train_dataset
                                   if f"_SPLIT{cg_int}" in ds["item_id"]]
            
            if cg_train_datasets:
                # Use the target shape to estimate size
                avg_size = sum(ds["target"].shape[1] for ds in cg_train_datasets) / len(cg_train_datasets)
                total_size = avg_size * self.data_module.n_splits
                
                cg_info = {
                    "continuity_group": cg,
                    "total_size": int(total_size),
                    "avg_dataset_size": int(avg_size),
                    "num_datasets": len(cg_train_datasets)
                }
                result["continuity_groups"].append(cg_info)
            else:
                logger.warning(f"Could not find datasets for continuity group {cg} (int: {cg_int})")
                
        # Count samples in each dataset
        result["train_samples"] = len(self.data_module.train_dataset)
        result["val_samples"] = len(self.data_module.val_dataset)
        result["test_samples"] = len(self.data_module.test_dataset)
        result["total_samples"] = result["train_samples"] + result["val_samples"] + result["test_samples"]
        
        return result
    
    def analyze_valid_windows(self) -> Dict:
        """
        Analyze the number of valid windows per group.
        
        Returns:
            Dictionary with window analysis information
        """
        logger.info("Analyzing valid windows")
        
        # Initialize samplers
        train_sampler = SequentialSampler(
            min_past=self.data_module.context_length, 
            min_future=self.data_module.prediction_length
        )
        val_sampler = ValidationSplitSampler(
            min_past=self.data_module.context_length, 
            min_future=self.data_module.prediction_length
        )
        
        self.samplers = {
            "train": train_sampler,
            "val": val_sampler
        }
        
        # Count valid windows
        window_counts = {
            "train": [],
            "val": []
        }
        
        # Analyze train dataset
        for ds_idx, ds in enumerate(self.data_module.train_dataset):
            a, b = train_sampler._get_bounds(ds["target"])
            valid_windows = b - a + 1
            window_counts["train"].append({
                "dataset_idx": ds_idx,
                "item_id": ds["item_id"],
                "target_shape": ds["target"].shape,
                "valid_windows": valid_windows,
                "start_idx": a,
                "end_idx": b
            })
        
        # Analyze validation dataset
        for ds_idx, ds in enumerate(self.data_module.val_dataset):
            a, b = val_sampler._get_bounds(ds["target"])
            valid_windows = b - a + 1
            window_counts["val"].append({
                "dataset_idx": ds_idx,
                "item_id": ds["item_id"],
                "target_shape": ds["target"].shape,
                "valid_windows": valid_windows,
                "start_idx": a,
                "end_idx": b
            })
        
        self.window_counts = window_counts
        
        # Calculate total valid windows
        total_train_windows = sum(item["valid_windows"] for item in window_counts["train"])
        total_val_windows = sum(item["valid_windows"] for item in window_counts["val"])
        
        # Calculate training steps
        batch_size = self.data_module.batch_size
        train_steps = np.ceil(total_train_windows / batch_size).astype(int)
        val_steps = np.ceil(total_val_windows / batch_size).astype(int)
        
        # Check if limit_train_batches is set
        limit_train_batches = self.config["trainer"].get("limit_train_batches")
        if limit_train_batches:
            train_steps = min(train_steps, limit_train_batches)
        
        return {
            "window_counts": window_counts,
            "total_train_windows": total_train_windows,
            "total_val_windows": total_val_windows,
            "batch_size": batch_size,
            "train_steps": train_steps,
            "val_steps": val_steps,
            "limit_train_batches": limit_train_batches
        }
    
    def analyze_continuity_groups(self) -> Dict:
        """
        Analyze the continuity groups in detail.
        
        Returns:
            Dictionary with continuity group analysis
        """
        logger.info("Analyzing continuity groups")
        
        cg_info = {}
        
        # Calculate estimated rows_per_split for each continuity group
        estimated_rows_per_split = {}
        for cg in self.data_module.continuity_groups:
            # Convert floating-point continuity group to integer for pattern matching
            cg_int = int(cg)
            
            # Find datasets for this continuity group
            cg_train_datasets = [ds for ds in self.data_module.train_dataset
                               if f"SPLIT{cg_int}" in ds["item_id"]]
            
            if not cg_train_datasets:
                # Try alternative pattern for per_turbine_target=True
                cg_train_datasets = [ds for ds in self.data_module.train_dataset
                                   if f"_SPLIT{cg_int}" in ds["item_id"]]
            
            if cg_train_datasets:
                # Use the target shape to estimate size
                avg_size = sum(ds["target"].shape[1] for ds in cg_train_datasets) / len(cg_train_datasets)
                # Divide by train_split to get the total size
                total_size = avg_size / self.data_module.train_split
                # Divide by n_splits to get rows_per_split
                rows_per_split = total_size / self.data_module.n_splits
                estimated_rows_per_split[cg] = int(rows_per_split)
            else:
                logger.warning(f"Could not find datasets for continuity group {cg} (int: {cg_int})")
                estimated_rows_per_split[cg] = 0
        
        for cg in self.data_module.continuity_groups:
            if cg not in estimated_rows_per_split or estimated_rows_per_split[cg] == 0:
                logger.warning(f"Skipping continuity group {cg} due to missing size information")
                continue
                
            # Calculate total size
            rows_per_split = estimated_rows_per_split[cg]
            total_size = rows_per_split * self.data_module.n_splits
            
            # Calculate train/val/test sizes
            train_size = int(self.data_module.train_split * rows_per_split)
            val_size = int(self.data_module.val_split * rows_per_split)
            test_size = int(self.data_module.test_split * rows_per_split)
            
            # Check if sizes are sufficient for context + prediction
            min_required = self.data_module.context_length + self.data_module.prediction_length
            train_sufficient = train_size >= min_required
            val_sufficient = val_size >= min_required
            test_sufficient = test_size >= min_required
            
            cg_info[str(cg)] = {
                "total_size": int(total_size),
                "rows_per_split": rows_per_split,
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size,
                "min_required_size": min_required,
                "train_sufficient": train_sufficient,
                "val_sufficient": val_sufficient,
                "test_sufficient": test_sufficient
            }
        
        self.continuity_group_info = cg_info
        return {"continuity_group_info": cg_info}
    
    def analyze_split_patterns(self) -> Dict:
        """
        Analyze the patterns in item_ids to understand how splits and continuity groups are represented.
        
        Returns:
            Dictionary with pattern analysis
        """
        logger.info("Analyzing split patterns in item_ids")
        
        # Extract all item_ids
        item_ids = [ds["item_id"] for ds in self.data_module.train_dataset]
        
        # Extract patterns
        split_pattern = re.compile(r'SPLIT(\d+)')
        turbine_pattern = re.compile(r'TURBINE(\d+)')
        
        # Extract all split numbers and turbine numbers
        split_numbers = []
        turbine_numbers = []
        
        for item_id in item_ids:
            split_match = split_pattern.search(item_id)
            if split_match:
                split_numbers.append(int(split_match.group(1)))
            
            turbine_match = turbine_pattern.search(item_id)
            if turbine_match:
                turbine_numbers.append(int(turbine_match.group(1)))
        
        # Get unique values
        unique_splits = sorted(set(split_numbers))
        unique_turbines = sorted(set(turbine_numbers))
        
        # Compare with continuity groups
        continuity_groups = [int(cg) for cg in self.data_module.continuity_groups]
        
        # Check if continuity groups match split numbers
        matches_splits = set(continuity_groups) == set(unique_splits)
        
        return {
            "unique_split_numbers": unique_splits,
            "unique_turbine_numbers": unique_turbines,
            "continuity_groups_as_ints": continuity_groups,
            "continuity_groups_match_splits": matches_splits,
            "sample_item_ids": item_ids[:5] if len(item_ids) > 5 else item_ids
        }
    
    def visualize_window_distribution(self) -> None:
        """Visualize the distribution of valid windows across datasets."""
        logger.info("Visualizing window distribution")
        
        # Extract window counts
        train_windows = [item["valid_windows"] for item in self.window_counts["train"]]
        val_windows = [item["valid_windows"] for item in self.window_counts["val"]]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        plt.subplot(2, 1, 1)
        plt.hist(train_windows, bins=20, alpha=0.7, label='Train')
        plt.axvline(x=1, color='r', linestyle='--', label='Single Window')
        plt.title('Distribution of Valid Windows per Dataset (Train)')
        plt.xlabel('Number of Valid Windows')
        plt.ylabel('Count')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(val_windows, bins=20, alpha=0.7, label='Validation')
        plt.axvline(x=1, color='r', linestyle='--', label='Single Window')
        plt.title('Distribution of Valid Windows per Dataset (Validation)')
        plt.xlabel('Number of Valid Windows')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.join(os.path.dirname(self.config_path), 'debug_outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'window_distribution.png'))
        logger.info(f"Saved window distribution visualization to {output_dir}/window_distribution.png")
    
    def visualize_dataset_structure(self) -> None:
        """Visualize the structure of the dataset."""
        logger.info("Visualizing dataset structure")
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot continuity group sizes
        cg_sizes = [info["total_size"] for info in self.continuity_group_info.values()]
        cg_labels = [f"CG {cg}" for cg in self.continuity_group_info.keys()]
        
        if cg_sizes:  # Only plot if we have data
            plt.subplot(2, 2, 1)
            plt.bar(cg_labels, cg_sizes)
            plt.title('Continuity Group Sizes')
            plt.xlabel('Continuity Group')
            plt.ylabel('Size (rows)')
            plt.xticks(rotation=45)
            
            # Plot train/val/test split sizes
            train_sizes = [info["train_size"] for info in self.continuity_group_info.values()]
            val_sizes = [info["val_size"] for info in self.continuity_group_info.values()]
            test_sizes = [info["test_size"] for info in self.continuity_group_info.values()]
            
            plt.subplot(2, 2, 2)
            width = 0.25
            x = np.arange(len(cg_labels))
            plt.bar(x - width, train_sizes, width, label='Train')
            plt.bar(x, val_sizes, width, label='Validation')
            plt.bar(x + width, test_sizes, width, label='Test')
            plt.axhline(y=self.data_module.context_length + self.data_module.prediction_length,
                       color='r', linestyle='--', label='Min Required')
            plt.title('Train/Val/Test Split Sizes')
            plt.xlabel('Continuity Group')
            plt.ylabel('Size (rows)')
            plt.xticks(x, cg_labels, rotation=45)
            plt.legend()
        else:
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, "No continuity group data available",
                    horizontalalignment='center', verticalalignment='center')
            plt.subplot(2, 2, 2)
            plt.text(0.5, 0.5, "No continuity group data available",
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot valid windows per dataset
        train_windows = [item["valid_windows"] for item in self.window_counts["train"]]
        
        plt.subplot(2, 2, 3)
        if train_windows:
            plt.hist(train_windows, bins=np.arange(min(train_windows), max(train_windows) + 1, 1))
            plt.title('Valid Windows per Dataset (Train)')
            plt.xlabel('Number of Valid Windows')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, "No window data available",
                    horizontalalignment='center', verticalalignment='center')
        
        # Plot context and prediction length
        plt.subplot(2, 2, 4)
        plt.bar(['Context Length', 'Prediction Length'], 
               [self.data_module.context_length, self.data_module.prediction_length])
        plt.title('Context and Prediction Length (steps)')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.join(os.path.dirname(self.config_path), 'debug_outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'dataset_structure.png'))
        logger.info(f"Saved dataset structure visualization to {output_dir}/dataset_structure.png")
    
    def generate_report(self) -> Dict:
        """
        Generate a comprehensive report with all analysis results.
        
        Returns:
            Dictionary with all analysis results
        """
        logger.info("Generating comprehensive report")
        
        # Add split pattern analysis to help debug continuity group issues
        split_patterns = self.analyze_split_patterns()
        
        report = {
            "context_length_analysis": self.analyze_context_length_precedence(),
            "dataset_structure": self.analyze_dataset_structure(),
            "valid_windows": self.analyze_valid_windows(),
            "continuity_groups": self.analyze_continuity_groups(),
            "split_patterns": split_patterns
        }
        
        # Generate visualizations
        self.visualize_window_distribution()
        self.visualize_dataset_structure()
        
        # Save report to JSON
        output_dir = os.path.join(os.path.dirname(self.config_path), 'debug_outputs')
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'data_analysis_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive report to {report_path}")
        
        return report
    
    def print_report_summary(self, report: Dict) -> None:
        """
        Print a summary of the report to the console.
        
        Args:
            report: The full report dictionary
        """
        logger.info("Printing report summary")
        
        # Context length analysis
        context_analysis = report["context_length_analysis"]
        print("\n" + "="*80)
        print("CONTEXT LENGTH ANALYSIS")
        print("="*80)
        print(f"Explicit context_length: {context_analysis['explicit_context_length_seconds']} seconds = {context_analysis['context_steps_from_explicit']} steps")
        print(f"Prediction length: {context_analysis['explicit_prediction_length_seconds']} seconds = {context_analysis['prediction_steps']} steps")
        print(f"Context length factor: {context_analysis['context_length_factor']}")
        if context_analysis['context_length_factor']:
            print(f"Factor-based context length: {context_analysis['context_steps_from_factor']} steps")
        print(f"Precedence: {context_analysis['precedence']}")
        print(f"Expected context steps: {context_analysis['expected_context_steps']}")
        print(f"Actual context steps: {context_analysis['actual_context_steps']}")
        if context_analysis['mismatch']:
            print("WARNING: Mismatch between expected and actual context steps!")
        
        # Dataset structure
        dataset_structure = report["dataset_structure"]
        print("\n" + "="*80)
        print("DATASET STRUCTURE")
        print("="*80)
        print(f"Total samples: {dataset_structure['total_samples']}")
        print(f"Train samples: {dataset_structure['train_samples']}")
        print(f"Validation samples: {dataset_structure['val_samples']}")
        print(f"Test samples: {dataset_structure['test_samples']}")
        print(f"Per-turbine target: {dataset_structure['per_turbine_target']}")
        print(f"Number of turbines: {len(dataset_structure['turbine_ids'])}")
        
        # Split patterns
        split_patterns = report["split_patterns"]
        print("\n" + "="*80)
        print("SPLIT PATTERNS ANALYSIS")
        print("="*80)
        print(f"Unique split numbers in item_ids: {split_patterns['unique_split_numbers']}")
        print(f"Unique turbine numbers in item_ids: {split_patterns['unique_turbine_numbers']}")
        print(f"Continuity groups as integers: {split_patterns['continuity_groups_as_ints']}")
        print(f"Continuity groups match split numbers: {split_patterns['continuity_groups_match_splits']}")
        print(f"Sample item_ids: {split_patterns['sample_item_ids']}")
        
        # Valid windows
        valid_windows = report["valid_windows"]
        print("\n" + "="*80)
        print("VALID WINDOWS ANALYSIS")
        print("="*80)
        print(f"Total train windows: {valid_windows['total_train_windows']}")
        print(f"Total validation windows: {valid_windows['total_val_windows']}")
        print(f"Batch size: {valid_windows['batch_size']}")
        print(f"Training steps: {valid_windows['train_steps']}")
        print(f"Validation steps: {valid_windows['val_steps']}")
        
        # Single window datasets
        single_window_count = sum(1 for item in valid_windows['window_counts']['train'] if item['valid_windows'] == 1)
        print(f"\nDatasets with only ONE valid window: {single_window_count} out of {len(valid_windows['window_counts']['train'])} ({single_window_count/len(valid_windows['window_counts']['train'])*100:.1f}%)")
        
        if single_window_count > 0:
            print("\nPossible reasons for single-window datasets:")
            print("1. Dataset size is too small relative to context_length + prediction_length")
            print("2. The split sizes (train_split, val_split, test_split) create segments that are too small")
            print("3. The context_length is too large compared to the available data")
            
            print("\nPossible solutions:")
            print("1. Decrease context_length or context_length_factor")
            print("2. Adjust train/val/test split ratios to ensure larger segments")
            print("3. Decrease prediction_length")
            print("4. Use a coarser frequency (e.g., 120s instead of 60s)")
            print("5. Combine multiple continuity groups")
        
        # Continuity groups
        print("\n" + "="*80)
        print("CONTINUITY GROUPS ANALYSIS")
        print("="*80)
        
        cg_table = []
        for cg, info in report["continuity_groups"]["continuity_group_info"].items():
            cg_table.append([
                cg,
                info["total_size"],
                info["train_size"],
                info["val_size"],
                info["test_size"],
                info["min_required_size"],
                "✓" if info["train_sufficient"] else "✗",
                "✓" if info["val_sufficient"] else "✗",
                "✓" if info["test_sufficient"] else "✗"
            ])
        
        if cg_table:
            print(tabulate(cg_table, headers=[
                "CG", "Total Size", "Train Size", "Val Size", "Test Size",
                "Min Required", "Train OK", "Val OK", "Test OK"
            ]))
        else:
            print("No continuity group information available.")
        
        # Visualizations
        print("\n" + "="*80)
        print("VISUALIZATIONS")
        print("="*80)
        output_dir = os.path.join(os.path.dirname(self.config_path), 'debug_outputs')
        print(f"Window distribution: {output_dir}/window_distribution.png")
        print(f"Dataset structure: {output_dir}/dataset_structure.png")
        print(f"Full report: {output_dir}/data_analysis_report.json")
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        # Make recommendations based on analysis
        if single_window_count / len(valid_windows['window_counts']['train']) > 0.5:
            print("CRITICAL: More than 50% of datasets have only one valid window!")
            print("This severely limits the model's ability to learn temporal patterns.")
            
            # Calculate a better context length
            avg_dataset_size = np.mean([info.get("train_size", 0) for info in report["continuity_groups"]["continuity_group_info"].values()])
            recommended_context_length = int(avg_dataset_size * 0.4)  # Use 40% of average dataset size
            recommended_prediction_length = int(avg_dataset_size * 0.2)  # Use 20% of average dataset size
            
            print(f"\nRecommended changes:")
            print(f"1. Reduce context_length from {context_analysis['actual_context_steps']} to approximately {recommended_context_length} steps")
            print(f"2. Consider reducing prediction_length from {context_analysis['prediction_steps']} to approximately {recommended_prediction_length} steps")
            print(f"3. If using context_length_factor, reduce it from {context_analysis['context_length_factor']} to approximately {recommended_context_length/context_analysis['prediction_steps']:.1f}")
            
        elif context_analysis['mismatch']:
            print("WARNING: There's a mismatch between expected and actual context length!")
            print(f"Expected: {context_analysis['expected_context_steps']} steps, Actual: {context_analysis['actual_context_steps']} steps")
            print("This could indicate a configuration issue or a bug in the code.")
            
        if valid_windows['train_steps'] < 10:
            print("\nWARNING: Very few training steps per epoch!")
            print(f"Current steps: {valid_windows['train_steps']}")
            print("Consider increasing batch size or adjusting dataset parameters to ensure more training steps.")
            
        print("\nFor more detailed analysis, please refer to the generated JSON report and visualizations.")

def main():
    """Main function to run the data debugging script."""
    parser = argparse.ArgumentParser(description="Debug data structure for wind forecasting")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, default="tactis", choices=["tactis", "informer", "autoformer", "spacetimeformer"], 
                        help="Model type to use for debugging")
    args = parser.parse_args()
    
    # Initialize debugger
    debugger = DataDebugger(args.config, args.model)
    
    # Initialize data module
    debugger.initialize_data_module()
    
    # Generate and print report
    report = debugger.generate_report()
    debugger.print_report_summary(report)
    
    logger.info("Data debugging completed successfully")

if __name__ == "__main__":
    main()