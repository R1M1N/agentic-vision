"""
YOLO Training System - replaces viseon train functionality
Complete training pipeline with Ultralytics YOLO
"""

import os
import json
import yaml
import shutil
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import subprocess
import threading
import time
from datetime import datetime
import mlflow
import mlflow.pytorch
import numpy as np

class YOLOTrainer:
    """
    YOLO Training System
    Provides complete training pipeline with experiment tracking
    """
    
    def __init__(self, config: Dict):
        """
        Initialize YOLO Trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mlflow_config = config.get('mlflow', {})
        self.training_config = config.get('training', {})
        
        # Training state
        self.current_run = None
        self.training_process = None
        self.training_log = []
        
        # Model configurations
        self.model_configs = {
            'yolov8n': {
                'model_path': 'yolov8n.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 16
            },
            'yolov8s': {
                'model_path': 'yolov8s.pt', 
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 16
            },
            'yolov8m': {
                'model_path': 'yolov8m.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 8
            },
            'yolov8l': {
                'model_path': 'yolov8l.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 4
            },
            'yolov8x': {
                'model_path': 'yolov8x.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 2
            },
            'yolov11n': {
                'model_path': 'yolo11n.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 16
            },
            'yolov11s': {
                'model_path': 'yolo11s.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 16
            },
            'yolov11m': {
                'model_path': 'yolo11m.pt',
                'epochs': 100,
                'imgsz': 640,
                'batch_size': 8
            }
        }
        
        # Initialize MLflow
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        tracking_uri = self.mlflow_config.get('tracking_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = self.mlflow_config.get('experiment_name', 'viseon_training')
        
        try:
            mlflow.get_experiment_by_name(experiment_name)
        except mlflow.exceptions.MlflowException:
            mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
    
    def start_training(self,
                      model_type: str = "yolov8n",
                      dataset_path: str = None,
                      dataset_version: str = None,
                      epochs: int = 100,
                      batch_size: int = None,
                      image_size: int = 640,
                      learning_rate: float = 0.01,
                      optimizer: str = "auto",
                      augment: bool = True,
                      save_period: int = 10,
                      project_name: str = "viseon_training",
                      experiment_name: str = None,
                      resume: bool = False,
                      pretrained: bool = True) -> Dict:
        """
        Start YOLO training (replaces viseon train)
        
        Args:
            model_type: Model type (yolov8n, yolov8s, yolov8m, etc.)
            dataset_path: Path to dataset
            dataset_version: DVC dataset version to use
            epochs: Number of training epochs
            batch_size: Batch size (auto-detected if None)
            image_size: Input image size
            learning_rate: Initial learning rate
            optimizer: Optimizer type
            augment: Use data augmentation
            save_period: Save model every N epochs
            project_name: MLflow project name
            experiment_name: MLflow experiment name
            resume: Resume training from checkpoint
            pretrained: Use pretrained weights
            
        Returns:
            Training configuration and start info
        """
        if experiment_name is None:
            experiment_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup training configuration
        training_config = self._prepare_training_config(
            model_type=model_type,
            dataset_path=dataset_path,
            dataset_version=dataset_version,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            augment=augment,
            save_period=save_period,
            project_name=project_name,
            experiment_name=experiment_name
        )
        
        # Start training process
        return self._start_training_process(training_config, resume, pretrained)
    
    def _prepare_training_config(self,
                                model_type: str,
                                dataset_path: str,
                                dataset_version: str,
                                epochs: int,
                                batch_size: int,
                                image_size: int,
                                learning_rate: float,
                                optimizer: str,
                                augment: bool,
                                save_period: int,
                                project_name: str,
                                experiment_name: str) -> Dict:
        """Prepare training configuration"""
        
        # Get model-specific config
        if model_type not in self.model_configs:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        base_config = self.model_configs[model_type].copy()
        
        # Use provided values or defaults
        config = {
            'model_type': model_type,
            'model_path': base_config['model_path'],
            'dataset_path': dataset_path,
            'dataset_version': dataset_version,
            'epochs': epochs,
            'batch_size': batch_size or base_config['batch_size'],
            'image_size': image_size,
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'augment': augment,
            'save_period': save_period,
            'project_name': project_name,
            'experiment_name': experiment_name,
            'device': 'auto',  # auto-detect GPU
            'workers': 8,
            'patience': 50,  # Early stopping patience
            'save': True,
            'save_json': True,
            'save_hub': False,
            'exist_ok': False,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 0,
            'resume': False
        }
        
        # Adjust batch size based on GPU memory if needed
        config['batch_size'] = self._optimize_batch_size(config['batch_size'], image_size)
        
        # Create YOLO configuration file
        self._create_dataset_yaml(config)
        
        return config
    
    def _optimize_batch_size(self, batch_size: int, image_size: int) -> int:
        """Optimize batch size based on available GPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Rough estimation: 8GB for 640x640, scale accordingly
                target_memory = 8 * 1024**3  # 8GB
                scaling_factor = target_memory / gpu_memory if gpu_memory > 0 else 1.0
                
                optimized_batch = int(batch_size * scaling_factor)
                optimized_batch = max(1, min(optimized_batch, 64))  # Reasonable bounds
                
                print(f"Optimized batch size: {batch_size} -> {optimized_batch}")
                return optimized_batch
        except ImportError:
            pass
        
        return batch_size
    
    def _create_dataset_yaml(self, config: Dict):
        """Create YOLO dataset configuration file"""
        if not config['dataset_path']:
            raise ValueError("Dataset path is required")
        
        # Create dataset YAML for YOLO
        dataset_config = {
            'train': f"{config['dataset_path']}/train/images",
            'val': f"{config['dataset_path']}/valid/images",
            'test': f"{config['dataset_path']}/test/images" if Path(f"{config['dataset_path']}/test/images").exists() else None,
            'nc': 0,  # Number of classes (will be detected)
            'names': []  # Class names (will be detected)
        }
        
        # Detect number of classes and names from label files
        nc, names = self._detect_dataset_classes(config['dataset_path'])
        dataset_config['nc'] = nc
        dataset_config['names'] = names
        
        # Save configuration
        config['dataset_yaml'] = f"{config['dataset_path']}/dataset.yaml"
        with open(config['dataset_yaml'], 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created dataset configuration: {config['dataset_yaml']}")
    
    def _detect_dataset_classes(self, dataset_path: Path) -> Tuple[int, List[str]]:
        """Detect number of classes and class names from dataset"""
        # This is a simplified implementation
        # Real implementation would parse label files to extract unique classes
        
        # Look for YOLO label files
        label_files = list(Path(dataset_path).rglob("*.txt"))
        
        classes = set()
        for label_file in label_files[:100]:  # Sample first 100 files
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            classes.add(class_id)
            except:
                continue
        
        # Create class names
        num_classes = len(classes) if classes else 1
        class_names = [f"class_{i}" for i in sorted(classes)] if classes else ["object"]
        
        return num_classes, class_names
    
    def _start_training_process(self, config: Dict, resume: bool, pretrained: bool) -> Dict:
        """Start the actual training process"""
        
        # Start MLflow run
        with mlflow.start_run(run_name=config['experiment_name']):
            # Log parameters
            mlflow.log_params({
                'model_type': config['model_type'],
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'image_size': config['image_size'],
                'learning_rate': config['learning_rate'],
                'optimizer': config['optimizer'],
                'augment': config['augment'],
                'dataset_path': config['dataset_path']
            })
            
            # Prepare training command
            cmd = self._build_training_command(config, resume, pretrained)
            
            print(f"Starting training with command: {' '.join(cmd)}")
            print(f"Model: {config['model_type']}")
            print(f"Dataset: {config['dataset_path']}")
            print(f"Epochs: {config['epochs']}")
            print(f"Batch Size: {config['batch_size']}")
            print(f"Image Size: {config['image_size']}")
            
            # Start training in background thread
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Log run info
            self.current_run = {
                'run_id': mlflow.active_run().info.run_id,
                'experiment_id': mlflow.active_run().info.experiment_id,
                'command': ' '.join(cmd),
                'config': config,
                'start_time': datetime.now().isoformat(),
                'status': 'running'
            }
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            return {
                'status': 'started',
                'mlflow_run_id': self.current_run['run_id'],
                'experiment_name': config['experiment_name'],
                'config': config,
                'command': ' '.join(cmd),
                'mlflow_ui_url': f"{self.mlflow_config.get('tracking_uri', 'http://localhost:5000')}/#/experiments/{self.current_run['experiment_id']}/runs/{self.current_run['run_id']}"
            }
    
    def _build_training_command(self, config: Dict, resume: bool, pretrained: bool) -> List[str]:
        """Build YOLO training command"""
        cmd = [
            'yolo', 'train',
            'model=' + config['model_path'],
            'data=' + config['dataset_yaml'],
            'epochs=' + str(config['epochs']),
            'batch=' + str(config['batch_size']),
            'imgsz=' + str(config['image_size']),
            'lr0=' + str(config['learning_rate']),
            'optimizer=' + config['optimizer'],
            'save_period=' + str(config['save_period']),
            'project=' + config['project_name'],
            'name=' + config['experiment_name'],
            'device=' + config['device'],
            'workers=' + str(config['workers']),
            'patience=' + str(config['patience']),
            'save=' + str(config['save']).lower(),
            'save_json=' + str(config['save_json']).lower(),
            'exist_ok=' + str(config['exist_ok']).lower(),
            'verbose=' + str(config['verbose']).lower(),
            'seed=' + str(config['seed']),
            'deterministic=' + str(config['deterministic']).lower(),
            'single_cls=' + str(config['single_cls']).lower(),
            'rect=' + str(config['rect']).lower(),
            'cos_lr=' + str(config['cos_lr']).lower()
        ]
        
        if config['augment']:
            cmd.append('hsv_h=0.015')
            cmd.append('hsv_s=0.7')
            cmd.append('hsv_v=0.4')
            cmd.append('degrees=0.0')
            cmd.append('translate=0.1')
            cmd.append('scale=0.5')
            cmd.append('shear=0.0')
            cmd.append('perspective=0.0')
            cmd.append('flipud=0.0')
            cmd.append('fliplr=0.5')
            cmd.append('mosaic=1.0')
            cmd.append('mixup=0.0')
            cmd.append('copy_paste=0.0')
        
        if resume:
            cmd.append('resume=true')
        elif not pretrained:
            cmd.append('pretrained=false')
        
        return cmd
    
    def _monitor_training(self):
        """Monitor training progress and log metrics"""
        if not self.training_process:
            return
        
        while self.training_process.poll() is None:
            try:
                output = self.training_process.stdout.readline()
                if output:
                    print(output.strip())
                    self.training_log.append(output.strip())
                    
                    # Parse metrics from output
                    self._parse_training_metrics(output)
                    
            except Exception as e:
                print(f"Error monitoring training: {e}")
                break
        
        # Training completed
        if self.current_run:
            self.current_run['status'] = 'completed'
            self.current_run['end_time'] = datetime.now().isoformat()
    
    def _parse_training_metrics(self, output: str):
        """Parse training metrics from YOLO output"""
        # This is a simplified parser
        # Real implementation would use more sophisticated parsing
        
        import re
        
        # Look for metrics like: Epoch [1/100]: loss=0.123, mAP50=0.456
        metric_patterns = {
            'epoch': r'Epoch \[(\d+)/(\d+)\]',
            'loss': r'loss=([0-9.]+)',
            'mAP50': r'mAP50=([0-9.]+)',
            'mAP50_95': r'mAP50-95=([0-9.]+)',
            'precision': r'precision=([0-9.]+)',
            'recall': r'recall=([0-9.]+)'
        }
        
        metrics = {}
        for metric_name, pattern in metric_patterns.items():
            match = re.search(pattern, output)
            if match:
                metrics[metric_name] = float(match.group(1))
        
        if metrics:
            # Log metrics to MLflow
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                print(f"Failed to log metrics: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current training status"""
        if not self.current_run:
            return {'status': 'no_active_training'}
        
        status = {
            'run_id': self.current_run['run_id'],
            'experiment_name': self.current_run['config']['experiment_name'],
            'status': self.current_run['status'],
            'start_time': self.current_run['start_time'],
            'model_type': self.current_run['config']['model_type'],
            'epochs': self.current_run['config']['epochs'],
            'dataset_path': self.current_run['config']['dataset_path'],
            'log_lines': len(self.training_log),
            'recent_logs': self.training_log[-10:] if self.training_log else []
        }
        
        if 'end_time' in self.current_run:
            status['end_time'] = self.current_run['end_time']
        
        if self.training_process:
            status['process_running'] = self.training_process.poll() is None
        
        return status
    
    def stop_training(self) -> bool:
        """Stop current training"""
        if self.training_process:
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=30)
                
                if self.current_run:
                    self.current_run['status'] = 'stopped'
                    self.current_run['end_time'] = datetime.now().isoformat()
                
                return True
            except subprocess.TimeoutExpired:
                self.training_process.kill()
                return True
            except Exception as e:
                print(f"Failed to stop training: {e}")
                return False
        
        return False
    
    def get_best_model(self) -> str:
        """Get path to best trained model"""
        if not self.current_run:
            return None
        
        # Find best model in project directory
        project_dir = Path(f"runs/detect/{self.current_run['config']['experiment_name']}")
        
        if project_dir.exists():
            best_model = project_dir / 'weights' / 'best.pt'
            if best_model.exists():
                return str(best_model)
            
            last_model = project_dir / 'weights' / 'last.pt'
            if last_model.exists():
                return str(last_model)
        
        return None
    
    def export_model(self,
                    model_path: str,
                    export_formats: List[str] = None,
                    imgsz: int = 640) -> Dict:
        """
        Export trained model to different formats
        
        Args:
            model_path: Path to trained model
            export_formats: Export formats ("onnx", "tensorrt", "coreml", "tflite")
            imgsz: Input image size for export
            
        Returns:
            Export results
        """
        if export_formats is None:
            export_formats = ['onnx', 'tensorrt']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        results = {
            'model_path': model_path,
            'export_formats': export_formats,
            'exports': []
        }
        
        for fmt in export_formats:
            try:
                cmd = [
                    'yolo', 'export',
                    f'model={model_path}',
                    f'format={fmt}',
                    f'imgsz={imgsz}'
                ]
                
                print(f"Exporting to {fmt}...")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Find exported file
                    exported_path = self._find_exported_file(model_path, fmt)
                    results['exports'].append({
                        'format': fmt,
                        'path': exported_path,
                        'status': 'success'
                    })
                else:
                    results['exports'].append({
                        'format': fmt,
                        'status': 'failed',
                        'error': result.stderr
                    })
                    
            except Exception as e:
                results['exports'].append({
                    'format': fmt,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def _find_exported_file(self, model_path: str, format: str) -> str:
        """Find exported model file"""
        model_path = Path(model_path)
        base_name = model_path.stem
        
        # Common export locations
        possible_paths = [
            model_path.parent / f"{base_name}.{format}",
            model_path.parent / "exports" / f"{base_name}.{format}",
            model_path.parent / f"{base_name}_{format}.{format}"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get_training_history(self) -> List[Dict]:
        """Get training history from MLflow"""
        try:
            experiment = mlflow.get_experiment_by_name(
                self.mlflow_config.get('experiment_name', 'viseon_training')
            )
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            history = []
            for _, run in runs.iterrows():
                history.append({
                    'run_id': run.run_id,
                    'experiment_id': run.experiment_id,
                    'status': run.status,
                    'start_time': run.start_time,
                    'end_time': run.end_time,
                    'metrics': run.to_dict()['metrics'] if 'metrics' in run.to_dict() else {},
                    'params': run.to_dict()['params'] if 'params' in run.to_dict() else {}
                })
            
            return history
            
        except Exception as e:
            print(f"Failed to get training history: {e}")
            return []
    
    def hyperparameter_tuning(self,
                             base_config: Dict,
                             param_grid: Dict) -> Dict:
        """
        Run hyperparameter tuning
        
        Args:
            base_config: Base training configuration
            param_grid: Parameters to tune
            
        Returns:
            Tuning results
        """
        # This is a simplified hyperparameter tuning implementation
        # Real implementation would use Optuna or similar
        
        tuning_results = {
            'base_config': base_config,
            'param_grid': param_grid,
            'runs': [],
            'best_run': None,
            'best_metrics': {}
        }
        
        # Generate parameter combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(itertools.product(*param_values))
        
        print(f"Running hyperparameter tuning with {len(combinations)} combinations")
        
        for i, combination in enumerate(combinations):
            config = base_config.copy()
            
            # Update parameters
            for param_name, param_value in zip(param_names, combination):
                config[param_name] = param_value
            
            # Run training
            try:
                result = self.start_training(
                    model_type=config.get('model_type', 'yolov8n'),
                    dataset_path=config.get('dataset_path'),
                    epochs=config.get('epochs', 50),  # Reduced for tuning
                    batch_size=config.get('batch_size'),
                    image_size=config.get('image_size', 640)
                )
                
                tuning_results['runs'].append({
                    'run_id': result['mlflow_run_id'],
                    'config': config,
                    'status': 'started'
                })
                
            except Exception as e:
                tuning_results['runs'].append({
                    'run_id': None,
                    'config': config,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return tuning_results