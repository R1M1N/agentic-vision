"""
Project Management System - replaces roboflow-python project functionality
Handles data upload, versioning, and project-level operations
"""

import os
import shutil
import json
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import cv2
import numpy as np
from .detections import Detections


class Project:
    """
    OpenSupervision Project - manages datasets and versions
    Replaces roboflow-python Project functionality
    """
    
    def __init__(self, name: str, description: str = "", config: Dict = None):
        """
        Initialize Project
        
        Args:
            name: Project name
            description: Project description
            config: Configuration dictionary
        """
        self.name = name
        self.description = description
        self.config = config or {}
        
        # Project paths
        self.base_path = Path(self.config.get('storage', {}).get('base_path', './data'))
        self.project_path = self.base_path / 'projects' / name
        self.raw_data_path = self.project_path / 'raw'
        self.processed_path = self.project_path / 'processed'
        self.versions_path = self.project_path / 'versions'
        self.annotations_path = self.project_path / 'annotations'
        self.models_path = self.project_path / 'models'
        
        # Create directory structure
        self._create_directories()
        
        # Project metadata
        self.metadata_file = self.project_path / 'project.yaml'
        self.samples_file = self.project_path / 'samples.json'
        self.metadata = self._load_or_create_metadata()
        
        # Current version
        self.current_version = None
    
    @property
    def project_dir(self):
        """Compatibility property for project directory"""
        return self.project_path
        
    def _create_directories(self):
        """Create project directory structure"""
        directories = [
            self.project_path,
            self.raw_data_path,
            self.processed_path,
            self.versions_path,
            self.annotations_path,
            self.models_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_or_create_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            metadata = {
                'name': self.name,
                'description': self.description,
                'created_at': str(Path().cwd()),
                'versions': [],
                'stats': {
                    'total_images': 0,
                    'total_videos': 0,
                    'classes': [],
                    'annotations_count': 0
                },
                'config': self.config
            }
            self._save_metadata(metadata)
            return metadata
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
    
    def upload_data(self, 
                   data_path: Union[str, Path],
                   include_labels: bool = False,
                   dataset_type: str = "auto") -> Dict:
        """
        Upload data to project (replaces roboflow-python upload)
        
        Args:
            data_path: Path to images/videos directory or file
            include_labels: Whether to include existing label files
            dataset_type: Type of dataset ("images", "videos", "mixed", "auto")
            
        Returns:
            Upload summary
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        # Determine dataset type
        if dataset_type == "auto":
            dataset_type = self._detect_dataset_type(data_path)
        
        summary = {
            'project_name': self.name,
            'dataset_type': dataset_type,
            'uploaded_files': [],
            'skipped_files': [],
            'total_size': 0,
            'file_count': 0
        }
        
        # Handle single file or directory
        if data_path.is_file():
            files = [data_path]
        else:
            files = self._get_dataset_files(data_path, dataset_type)
        
        # Upload files
        for file_path in files:
            try:
                uploaded_file = self._upload_single_file(file_path, include_labels)
                summary['uploaded_files'].append(uploaded_file)
                summary['total_size'] += uploaded_file['size']
                summary['file_count'] += 1
            except Exception as e:
                summary['skipped_files'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Update project metadata
        self._update_project_stats(summary)
        self._save_metadata(self.metadata)
        
        return summary
    
    def _detect_dataset_type(self, data_path: Path) -> str:
        """Detect dataset type from file extensions"""
        files = list(data_path.iterdir())
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        image_count = sum(1 for f in files if f.is_file() and f.suffix.lower() in image_extensions)
        video_count = sum(1 for f in files if f.is_file() and f.suffix.lower() in video_extensions)
        
        if image_count > 0 and video_count == 0:
            return "images"
        elif video_count > 0 and image_count == 0:
            return "videos"
        elif image_count > 0 and video_count > 0:
            return "mixed"
        else:
            return "unknown"
    
    def _get_dataset_files(self, data_path: Path, dataset_type: str) -> List[Path]:
        """Get dataset files based on type"""
        files = []
        
        if dataset_type == "images":
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            files = [f for f in data_path.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions]
        
        elif dataset_type == "videos":
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            files = [f for f in data_path.iterdir() 
                    if f.is_file() and f.suffix.lower() in video_extensions]
        
        elif dataset_type == "mixed":
            all_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp',
                            '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
            files = [f for f in data_path.iterdir() 
                    if f.is_file() and f.suffix.lower() in all_extensions]
        
        return sorted(files)
    
    def _upload_single_file(self, file_path: Path, include_labels: bool) -> Dict:
        """Upload single file to project"""
        # Create unique filename with hash
        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()[:8]
        file_ext = file_path.suffix
        new_name = f"{file_path.stem}_{file_hash}{file_ext}"
        
        # Copy file to raw data directory
        target_path = self.raw_data_path / new_name
        shutil.copy2(file_path, target_path)
        
        # Generate metadata
        file_size = file_path.stat().st_size
        file_info = {
            'original_name': file_path.name,
            'filename': new_name,
            'path': str(target_path),
            'size': file_size,
            'hash': file_hash,
            'type': file_path.suffix.lower(),
            'uploaded_at': str(Path().cwd()),
            'width': None,
            'height': None,
            'has_labels': False,
            'labels_path': None
        }
        
        # Get image/video dimensions
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            img = cv2.imread(str(file_path))
            if img is not None:
                file_info['width'] = img.shape[1]
                file_info['height'] = img.shape[0]
        
        # Check for label files
        if include_labels:
            label_path = self._find_label_file(file_path)
            if label_path:
                target_label_path = self.annotations_path / f"{new_name}_labels.txt"
                shutil.copy2(label_path, target_label_path)
                file_info['has_labels'] = True
                file_info['labels_path'] = str(target_label_path)
        
        return file_info
    
    def _find_label_file(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file"""
        label_extensions = ['.txt', '.json', '.xml', '.csv']
        label_names = [
            image_path.stem + ext for ext in label_extensions
        ]
        
        # Check in same directory
        for label_name in label_names:
            label_path = image_path.parent / label_name
            if label_path.exists():
                return label_path
        
        # Check in common label directories
        label_dirs = ['labels', 'annotations', 'yolo_labels', 'coco_annotations']
        for label_dir in label_dirs:
            for label_name in label_names:
                label_path = image_path.parent / label_dir / label_name
                if label_path.exists():
                    return label_path
        
        return None
    
    def _update_project_stats(self, upload_summary: Dict):
        """Update project statistics"""
        self.metadata['stats']['total_files'] = (
            self.metadata['stats'].get('total_files', 0) + upload_summary['file_count']
        )
        
        # Update file type counts
        if upload_summary['dataset_type'] == 'images':
            self.metadata['stats']['total_images'] = (
                self.metadata['stats'].get('total_images', 0) + upload_summary['file_count']
            )
        elif upload_summary['dataset_type'] == 'videos':
            self.metadata['stats']['total_videos'] = (
                self.metadata['stats'].get('total_videos', 0) + upload_summary['file_count']
            )
    
    def create_version(self, 
                      version_name: str,
                      description: str = "",
                      include_raw: bool = True,
                      include_annotations: bool = True) -> Dict:
        """
        Create a new dataset version
        
        Args:
            version_name: Name of the version
            description: Version description
            include_raw: Include raw data in version
            include_annotations: Include annotations in version
            
        Returns:
            Version information
        """
        version_path = self.versions_path / version_name
        
        # Create version directory structure
        version_structure = {
            'data': version_path / 'data',
            'labels': version_path / 'labels',
            'images': version_path / 'images',
            'metadata': version_path / 'metadata'
        }
        
        for path in version_structure.values():
            path.mkdir(parents=True, exist_ok=True)
        
        # Copy data files
        copied_files = []
        
        # Copy images
        for file_info in self._get_sample_list():
            if file_info['type'] in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                source = Path(file_info['path'])
                target = version_structure['images'] / file_info['filename']
                if include_raw and source.exists():
                    shutil.copy2(source, target)
                    copied_files.append(file_info['filename'])
            
            # Copy labels if available and requested
            if include_annotations and file_info.get('has_labels', False):
                source = Path(file_info['labels_path'])
                target = version_structure['labels'] / f"{file_info['filename']}_labels.txt"
                if source.exists():
                    shutil.copy2(source, target)
        
        # Create version metadata
        version_metadata = {
            'name': version_name,
            'description': description,
            'created_at': str(Path().cwd()),
            'parent_project': self.name,
            'files_count': len(copied_files),
            'include_raw': include_raw,
            'include_annotations': include_annotations,
            'files': copied_files,
            'config': {
                'image_format': 'jpg',
                'annotation_format': 'yolo',
                'image_size': [640, 640],
                'augmentation': False
            }
        }
        
        # Save version metadata
        metadata_file = version_structure['metadata'] / 'version.yaml'
        with open(metadata_file, 'w') as f:
            yaml.dump(version_metadata, f, default_flow_style=False)
        
        # Update project metadata
        self.metadata['versions'].append({
            'name': version_name,
            'description': description,
            'path': str(version_path),
            'created_at': version_metadata['created_at'],
            'files_count': len(copied_files)
        })
        
        self._save_metadata(self.metadata)
        self.current_version = version_name
        
        return version_metadata
    
    def _get_sample_list(self) -> List[Dict]:
        """Get list of all samples in project"""
        if self.samples_file.exists():
            with open(self.samples_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_dataset_info(self, version_name: str = None) -> Dict:
        """
        Get dataset information
        
        Args:
            version_name: Specific version to get info for
            
        Returns:
            Dataset information
        """
        if version_name:
            version_path = self.versions_path / version_name
            metadata_file = version_path / 'metadata' / 'version.yaml'
            
            if not metadata_file.exists():
                raise FileNotFoundError(f"Version {version_name} not found")
            
            with open(metadata_file, 'r') as f:
                version_info = yaml.safe_load(f)
            
            return version_info
        else:
            # Return project-level info
            info = {
                'name': self.name,
                'description': self.description,
                'stats': self.metadata['stats'],
                'versions': self.metadata['versions'],
                'current_version': self.current_version,
                'config': self.metadata.get('config', {})
            }
            return info
    
    def download_dataset(self, 
                        version_name: str,
                        output_path: Union[str, Path],
                        format: str = "yolo") -> Dict:
        """
        Download dataset in specified format
        
        Args:
            version_name: Version to download
            output_path: Output directory path
            format: Output format ("yolo", "coco", "pascal_voc")
            
        Returns:
            Download summary
        """
        version_info = self.get_dataset_info(version_name)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'version': version_name,
            'format': format,
            'output_path': str(output_path),
            'files_downloaded': [],
            'annotations_converted': []
        }
        
        version_path = self.versions_path / version_name
        
        if format == "yolo":
            # YOLO format: images/ and labels/ directories
            images_dir = output_path / 'images'
            labels_dir = output_path / 'labels'
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # Copy images
            for file_info in version_info['files']:
                source = version_path / 'images' / file_info
                target = images_dir / file_info
                if source.exists():
                    shutil.copy2(source, target)
                    summary['files_downloaded'].append(file_info)
        
        elif format == "coco":
            # COCO format: single JSON with all annotations
            coco_data = {
                'images': [],
                'annotations': [],
                'categories': []
            }
            
            # This would require parsing the label files and converting to COCO format
            # Implementation would depend on the specific annotation format used
        
        # Save conversion info
        info_file = output_path / 'download_info.yaml'
        with open(info_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        return summary
    
    def export_annotations(self, 
                          output_path: Union[str, Path],
                          format: str = "yolo",
                          version_name: str = None) -> Dict:
        """
        Export annotations in specified format
        
        Args:
            output_path: Output directory for annotations
            format: Output format ("yolo", "coco", "pascal_voc")
            version_name: Specific version to export from
            
        Returns:
            Export summary
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if version_name is None:
            version_name = self.current_version
        
        version_path = self.versions_path / version_name
        annotations_path = version_path / 'labels'
        
        summary = {
            'version': version_name,
            'format': format,
            'annotations_exported': [],
            'errors': []
        }
        
        if format == "yolo":
            # Export in YOLO format
            labels_dir = output_path
            labels_dir.mkdir(exist_ok=True)
            
            for label_file in annotations_path.glob('*_labels.txt'):
                try:
                    target = labels_dir / label_file.name
                    shutil.copy2(label_file, target)
                    summary['annotations_exported'].append(label_file.name)
                except Exception as e:
                    summary['errors'].append(f"Failed to export {label_file.name}: {str(e)}")
        
        elif format == "coco":
            # Convert to COCO format
            coco_annotations = self._convert_to_coco_format(annotations_path)
            coco_file = output_path / 'annotations.json'
            
            with open(coco_file, 'w') as f:
                json.dump(coco_annotations, f, indent=2)
            
            summary['annotations_exported'].append('annotations.json')
        
        # Save export info
        info_file = output_path / 'export_info.yaml'
        with open(info_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        return summary
    
    def _convert_to_coco_format(self, labels_path: Path) -> Dict:
        """Convert labels to COCO format"""
        # This is a simplified implementation
        # Real implementation would parse YOLO/TXT files and convert to COCO JSON
        
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add dummy categories - real implementation would extract from data
        coco_data['categories'] = [
            {'id': 1, 'name': 'object', 'supercategory': 'none'}
        ]
        
        return coco_data
    
    def delete_version(self, version_name: str) -> bool:
        """
        Delete a dataset version
        
        Args:
            version_name: Name of version to delete
            
        Returns:
            True if successful
        """
        version_path = self.versions_path / version_name
        
        if not version_path.exists():
            raise FileNotFoundError(f"Version {version_name} not found")
        
        # Remove version directory
        shutil.rmtree(version_path)
        
        # Remove from metadata
        self.metadata['versions'] = [
            v for v in self.metadata['versions'] 
            if v['name'] != version_name
        ]
        
        self._save_metadata(self.metadata)
        
        # Reset current version if it was deleted
        if self.current_version == version_name:
            self.current_version = None
        
        return True
    
    def list_versions(self) -> List[Dict]:
        """List all versions in project"""
        return self.metadata['versions']
    
    def get_project_stats(self) -> Dict:
        """Get comprehensive project statistics"""
        stats = self.metadata['stats'].copy()
        stats['versions_count'] = len(self.metadata['versions'])
        stats['project_size'] = self._calculate_project_size()
        stats['last_updated'] = self.metadata.get('created_at', '')
        
        return stats
    
    def _calculate_project_size(self) -> int:
        """Calculate total project size in bytes"""
        total_size = 0
        
        # Calculate raw data size
        for file_path in self.raw_data_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        # Calculate processed data size
        for file_path in self.processed_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        # Calculate versions size
        for version in self.metadata['versions']:
            version_path = Path(version['path'])
            if version_path.exists():
                for file_path in version_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        return total_size
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Project(name='{self.name}', versions={len(self.metadata['versions'])}, files={self.metadata['stats'].get('total_files', 0)})"