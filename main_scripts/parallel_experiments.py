"""
Parallel Experiments Runner

This script runs multiple experiments in parallel with different hyperparameters,
managing GPU memory usage and logging stdout/stderr to organized files.

Usage:
    python -m main_scripts.parallel_experiments

The script will:
1. Generate all hyperparameter combinations using itertools.product
2. Monitor GPU memory and start new experiments when memory is available (>20% free)
3. Log each experiment's stdout/stderr to organized files in output/parallel_experiments/


To start the experiments:

CUDA_VISIBLE_DEVICES=0 python -m main_scripts.parallel_experiments
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
import queue

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")
    print("Will not be able to monitor GPU memory.")


class GPUMonitor:
    """Monitor GPU memory usage using NVIDIA Management Library."""
    
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.physical_gpu_index = self._get_physical_gpu_index(gpu_index)
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_index)
                self.available = True
                print(f"GPU Monitor initialized: Logical GPU {gpu_index} -> Physical GPU {self.physical_gpu_index}")
                # Print initial GPU info
                self._print_gpu_info()
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}")
                self.available = False
        else:
            self.available = False
    
    def _get_physical_gpu_index(self, logical_index: int) -> int:
        """
        Convert logical GPU index (from CUDA_VISIBLE_DEVICES) to physical GPU index.
        
        Args:
            logical_index: The logical GPU index (0, 1, 2, etc. as seen by PyTorch)
            
        Returns:
            Physical GPU index for NVML
        """
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        if cuda_visible is None:
            # No CUDA_VISIBLE_DEVICES set, logical == physical
            return logical_index
        
        # Parse CUDA_VISIBLE_DEVICES
        visible_devices = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
        
        if not visible_devices:
            return logical_index
        
        # Map logical to physical
        if logical_index < len(visible_devices):
            physical_index = visible_devices[logical_index]
            return physical_index
        else:
            print(f"Warning: Logical GPU {logical_index} not available. Using GPU 0.")
            return visible_devices[0] if visible_devices else 0
    
    def _print_gpu_info(self):
        """Print GPU information for debugging."""
        if not self.available:
            return
        
        try:
            name = pynvml.nvmlDeviceGetName(self.handle)
            # Handle both string and bytes
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            total_gb = mem_info.total / (1024 ** 3)
            used_gb = mem_info.used / (1024 ** 3)
            free_gb = mem_info.free / (1024 ** 3)
            print(f"GPU: {name}")
            print(f"Memory: {used_gb:.2f}GB used / {total_gb:.2f}GB total ({free_gb:.2f}GB free)")
        except Exception as e:
            print(f"Could not get GPU info: {e}")
    
    def get_memory_info(self) -> Tuple[float, float, float]:
        """
        Get GPU memory information.
        
        Returns:
            Tuple of (used_memory_gb, total_memory_gb, free_percentage)
        """
        if not self.available:
            return 0.0, 0.0, 100.0
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = mem_info.used / (1024 ** 3)
            total_gb = mem_info.total / (1024 ** 3)
            free_percent = (mem_info.free / mem_info.total) * 100
            return used_gb, total_gb, free_percent
        except Exception as e:
            print(f"Warning: Error getting GPU memory: {e}")
            return 0.0, 0.0, 100.0
    
    def has_sufficient_memory(self, min_free_percent: float = 20.0) -> bool:
        """Check if GPU has sufficient free memory."""
        _, _, free_percent = self.get_memory_info()
        return free_percent >= min_free_percent
    
    def __del__(self):
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ExperimentRunner:
    """Manages parallel experiment execution with GPU memory monitoring."""
    
    def __init__(self, 
                 output_dir: str = "output/parallel_experiments",
                 min_free_memory_percent: float = 20.0,
                 check_interval: float = 30.0,
                 gpu_index: int = 0,
                 max_parallel_without_monitoring: int = 1):
        """
        Initialize the experiment runner.
        
        Args:
            output_dir: Base directory for experiment logs
            min_free_memory_percent: Minimum free GPU memory (%) before starting new experiments
            check_interval: How often to check GPU memory (seconds)
            gpu_index: Which GPU to monitor (default: 0)
            max_parallel_without_monitoring: Maximum parallel experiments if GPU monitoring unavailable
        """
        self.output_dir = output_dir
        self.min_free_memory_percent = min_free_memory_percent
        self.check_interval = check_interval
        self.max_parallel_without_monitoring = max_parallel_without_monitoring
        self.gpu_monitor = GPUMonitor(gpu_index)
        self.running_processes: List[Tuple[subprocess.Popen, str, Any]] = []
        self.experiment_queue = queue.Queue()
        
        # Warn if GPU monitoring is not available
        if not self.gpu_monitor.available:
            print("\n" + "!"*80)
            print("WARNING: GPU monitoring not available!")
            print(f"Will run maximum {max_parallel_without_monitoring} experiment(s) at a time.")
            print("Install nvidia-ml-py for proper GPU memory monitoring:")
            print("  pip install nvidia-ml-py")
            print("!"*80 + "\n")
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_experiments(self, hyperparams: Dict[str, List]) -> List[Dict]:
        """
        Generate all experiment combinations from hyperparameters.
        
        Args:
            hyperparams: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of experiment configurations (each a dict)
        """
        keys = list(hyperparams.keys())
        values = list(hyperparams.values())
        
        experiments = []
        for combination in product(*values):
            exp_config = dict(zip(keys, combination))
            experiments.append(exp_config)
        
        return experiments
    
    def create_log_path(self, exp_config: Dict) -> Tuple[str, str]:
        """
        Create the log file path for an experiment.
        
        Args:
            exp_config: Experiment configuration dictionary
            
        Returns:
            Tuple of (log_file_path, log_directory)
        """
        # Extract category for subdirectory
        category = exp_config.get('categories', 'unknown')
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Build filename from hyperparameters (excluding category)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create readable param string (exclude category and long paths)
        param_parts = []
        exclude_keys = {'categories', 'dataframe_path', 'dataframe_path_original', 
                       'save_dir', 'device', 'gemini_logo_mask_path', 'model_path'}
        
        for key, value in sorted(exp_config.items()):
            if key not in exclude_keys:
                # Shorten common parameter names
                short_key = key.replace('model_type', 'mt').replace('backbone', 'bb') \
                              .replace('lambda_', 'lam').replace('n_per_type', 'npt') \
                              .replace('seed', 's')
                param_parts.append(f"{short_key}={value}")
        
        param_str = "_".join(param_parts)
        
        # Create filename: params_timestamp.log
        log_filename = f"{param_str}_{timestamp}.log"
        log_path = os.path.join(category_dir, log_filename)
        
        return log_path, category_dir
    
    def build_command(self, exp_config: Dict) -> List[str]:
        """
        Build the command to run an experiment.
        
        Args:
            exp_config: Experiment configuration dictionary
            
        Returns:
            Command as list of strings
        """
        cmd = [sys.executable, "-m", "main_scripts.cbm"]
        
        for key, value in exp_config.items():
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def start_experiment(self, exp_config: Dict) -> Optional[Tuple[subprocess.Popen, Any]]:
        """
        Start a single experiment as a subprocess.
        
        Args:
            exp_config: Experiment configuration dictionary
            
        Returns:
            Tuple of (Popen object, log_file handle) or None if failed to start
        """
        log_path, _ = self.create_log_path(exp_config)
        cmd = self.build_command(exp_config)
        
        print(f"\n{'='*80}")
        print("Starting experiment:")
        print(f"  Category: {exp_config.get('categories', 'N/A')}")
        print(f"  Config: {exp_config}")
        print(f"  Log file: {log_path}")
        print(f"{'='*80}\n")
        
        try:
            # Open log file for both stdout and stderr
            log_file = open(log_path, 'w', buffering=1)  # Line buffering
            
            # Write header to log file
            log_file.write(f"Experiment started at: {datetime.now().isoformat()}\n")
            log_file.write(f"Command: {' '.join(cmd)}\n")
            log_file.write("Configuration:\n")
            for key, value in exp_config.items():
                log_file.write(f"  {key}: {value}\n")
            log_file.write(f"\n{'='*80}\n\n")
            log_file.flush()
            
            # Set environment to force unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line buffered
                env=env
            )
            
            return process, log_file
            
        except Exception as e:
            print(f"Error starting experiment: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup_finished_processes(self):
        """Remove finished processes from the running list and close their log files."""
        still_running = []
        
        for process, log_info, log_file in self.running_processes:
            if process.poll() is not None:  # Process has finished
                returncode = process.returncode
                print(f"\nExperiment finished: {log_info}")
                print(f"  Return code: {returncode}")
                
                if returncode != 0:
                    if returncode == -15:
                        print("  ⚠ Process was terminated (SIGTERM)")
                    elif returncode < 0:
                        print(f"  ⚠ Process killed by signal {-returncode}")
                    else:
                        print("  ✗ Process exited with error")
                else:
                    print("  ✓ Success")
                
                # Close the log file
                try:
                    log_file.close()
                except Exception:
                    pass
            else:
                still_running.append((process, log_info, log_file))
        
        self.running_processes = still_running
    
    def run_experiments(self, experiments: List[Dict]):
        """
        Run all experiments, managing parallelism based on GPU memory.
        
        Args:
            experiments: List of experiment configurations
        """
        print(f"\nTotal experiments to run: {len(experiments)}")
        print(f"GPU memory threshold: {self.min_free_memory_percent}% free")
        print(f"Check interval: {self.check_interval} seconds")
        
        # Add all experiments to queue
        for exp in experiments:
            self.experiment_queue.put(exp)
        
        while not self.experiment_queue.empty() or self.running_processes:
            # Clean up finished processes
            self.cleanup_finished_processes()
            
            # Check GPU memory and start new experiments if possible
            if not self.experiment_queue.empty():
                used_gb, total_gb, free_percent = self.gpu_monitor.get_memory_info()
                
                print(f"\nGPU Memory: {used_gb:.2f}GB / {total_gb:.2f}GB used ({100-free_percent:.1f}% used, {free_percent:.1f}% free)")
                print(f"Running experiments: {len(self.running_processes)}")
                print(f"Queued experiments: {self.experiment_queue.qsize()}")
                
                # Determine if we can start a new experiment
                can_start = False
                
                if self.gpu_monitor.available:
                    # Use GPU memory monitoring
                    can_start = self.gpu_monitor.has_sufficient_memory(self.min_free_memory_percent)
                    if not can_start:
                        print(f"Waiting for more GPU memory (need {self.min_free_memory_percent}% free)...")
                else:
                    # Fallback: limit based on number of running experiments
                    can_start = len(self.running_processes) < self.max_parallel_without_monitoring
                    if not can_start:
                        print(f"Waiting for running experiments to finish (limit: {self.max_parallel_without_monitoring})...")
                
                if can_start:
                    exp_config = self.experiment_queue.get()
                    result = self.start_experiment(exp_config)
                    
                    if result:
                        process, log_file = result
                        log_info = f"Category: {exp_config.get('categories', 'N/A')}, Seed: {exp_config.get('seed', 'N/A')}"
                        self.running_processes.append((process, log_info, log_file))
                    else:
                        # Put it back in the queue if it failed to start
                        self.experiment_queue.put(exp_config)
                else:
                    print(f"Waiting for more GPU memory (need {self.min_free_memory_percent}% free)...")
            
            # Wait before checking again
            time.sleep(self.check_interval)
        
        print(f"\n{'='*80}")
        print("All experiments completed!")
        print(f"Logs saved in: {self.output_dir}")
        print(f"{'='*80}\n")


def main():
    """Main function to configure and run parallel experiments."""
    
    # =========================================================================
    # CONFIGURATION: Define your hyperparameter grid here
    # =========================================================================
    
    hyperparameters = {
        'mode': ['train'],
        'categories': ['hazelnut', 'bottle', 'cable'],  # Add more categories as needed
        'model_type': ['joint'],
        'backbone': ['mobilenet_v2'],
        'seed': [0, 1, 2],
        'device': ['cuda'],
        'lambda_': [0.55],
        'n_per_type': [1, 2, 3],
        'epochs': [100],
        
        # Paths - these should be templates that include {category}
        'dataframe_path': ['cbm_vad/cbm_data/mvtec/{category}_dataset_automated_gen_anomalies.csv'],
        'dataframe_path_original': ['cbm_vad/cbm_data/mvtec/{category}_dataset_automated_gen_concepts.csv'],
        'save_dir': ['cbm_vad/cbm_models/mvtec_weakly/{category}_models/cont1/seed_{seed}'],
        'model_path': ['cbm_vad/cbm_models/backbones/fine-tuned-mobilenet.pth'],
        'gemini_logo_mask_path': ['datasets/gemini_logo_mask.png'],
        
        # Boolean flags
        'use_concepts': [True],
        'use_gen_anomalies': [True],
        'contaminate': [True],
    }
    
    # =========================================================================
    # Initialize experiment runner
    # =========================================================================
    
    runner = ExperimentRunner(
        output_dir="output/parallel_experiments",
        min_free_memory_percent=5.0,  # Wait until 20% GPU memory is free
        check_interval=2.0,  # Check every 2 seconds
        gpu_index=0,  # Use first GPU
        max_parallel_without_monitoring=20  # Safety: only 1 experiment at a time if no GPU monitoring
    )
    
    # Generate all experiment combinations
    experiments = runner.generate_experiments(hyperparameters)
    
    # Substitute templates in paths
    processed_experiments = []
    for exp in experiments:
        exp_copy = exp.copy()
        category = exp_copy.get('categories', '')
        seed = exp_copy.get('seed', 0)
        
        # Replace {category} and {seed} placeholders in paths
        for key in ['dataframe_path', 'dataframe_path_original', 'save_dir']:
            if key in exp_copy and isinstance(exp_copy[key], str):
                exp_copy[key] = exp_copy[key].format(category=category, seed=seed)
        
        processed_experiments.append(exp_copy)
    
    # Run all experiments
    runner.run_experiments(processed_experiments)


if __name__ == "__main__":
    main()

