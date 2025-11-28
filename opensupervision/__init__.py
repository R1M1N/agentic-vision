"""
OpenSupervision: A complete open-source computer vision platform
A sovereign alternative to Roboflow ecosystem using only open-source components
"""

# Core imports (always available)
from .core.detections import Detections
from .core.project import Project
from .tracking.object_tracker import ObjectTracker

# Conditional imports (require additional dependencies)
try:
    from .annotation.cvat_integration import CVATAnnotate
    CVATAnnotate_available = True
except ImportError:
    CVATAnnotate_available = False

try:
    from .training.yolo_trainer import YOLOTrainer
    YOLOTrainer_available = True
except ImportError:
    YOLOTrainer_available = False

try:
    from .inference.server import InferenceServer
    InferenceServer_available = True
except ImportError:
    InferenceServer_available = False

try:
    from .versioning.dvc_manager import DVCManager
    DVCManager_available = True
except ImportError:
    DVCManager_available = False

try:
    from .storage.storage_manager import StorageManager
    StorageManager_available = True
except ImportError:
    StorageManager_available = False

try:
    from .visualization.fiftyone_integration import FiftyOneManager
    FiftyOneManager_available = True
except ImportError:
    FiftyOneManager_available = False

__version__ = "1.0.0"

# Build __all__ list with only available components
__all__ = ["Detections", "Project", "ObjectTracker"]

if CVATAnnotate_available:
    __all__.append("CVATAnnotate")

if YOLOTrainer_available:
    __all__.append("YOLOTrainer")

if InferenceServer_available:
    __all__.append("InferenceServer")

if DVCManager_available:
    __all__.append("DVCManager")

if StorageManager_available:
    __all__.append("StorageManager")

if FiftyOneManager_available:
    __all__.append("FiftyOneManager")

class OpenSupervision:
    """
    Main OpenSupervision platform - replaces roboflow-python SDK
    Provides unified interface to all components
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize OpenSupervision platform
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components when requested (lazy loading)
        self.project = None
        self.tracker = None
        self.cvat = None
        self.trainer = None
        self.server = None
        self.dvc = None
        self.storage = None
        self.fiftyone = None
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        import yaml
        if config_path:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self):
        """Default configuration"""
        return {
            'storage': {
                'type': 'local',  # 'local' or 'minio'
                'base_path': './data',
                'minio': {
                    'endpoint': 'localhost:9000',
                    'access_key': 'minioadmin',
                    'secret_key': 'minioadmin',
                    'secure': False
                }
            },
            'fiftyone': {
                'port': 5151,
                'mongodb': 'mongodb://localhost:27017'
            },
            'cvat': {
                'url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'password'
            },
            'mlflow': {
                'port': 5000,
                'tracking_uri': 'http://localhost:5000'
            },
            'inference': {
                'port': 8000,
                'model_path': './models',
                'batch_size': 8
            },
            'tracking': {
                'algorithm': 'bytetrack',  # bytetrack, deepsort, botsort
                'max_age': 30,
                'min_hits': 3
            }
        }
    
    def create_project(self, name: str, description: str = ""):
        """
        Create a new project (replaces roboflow-python create_project)
        
        Args:
            name: Project name
            description: Project description
            
        Returns:
            Project instance
        """
        self.project = Project(name, description, self.config)
        return self.project
    
    def upload_data(self, data_path: str, project_name: str = None):
        """
        Upload data to project (replaces roboflow-python upload)
        
        Args:
            data_path: Path to images/video directory
            project_name: Target project name
        """
        if not self.project:
            project_name = project_name or "default"
            self.create_project(project_name)
        
        self.project.upload_data(data_path)
    
    def annotate(self, subset=None):
        """
        Start annotation workflow (replaces roboflow-python annotate)
        
        Args:
            subset: Optional subset of data to annotate
        """
        self.cvat = CVATAnnotate(self.config['cvat'])
        return self.cvat
    
    def train(self, model_type: str = "yolov8n", dataset_version: str = None):
        """
        Start training workflow (replaces roboflow-python train)
        
        Args:
            model_type: Model type (yolov8n, yolov8s, yolov11m, etc.)
            dataset_version: DVC dataset version to use
        """
        self.trainer = YOLOTrainer(self.config)
        return self.trainer.start_training(model_type, dataset_version)
    
    def deploy(self, model_path: str = None):
        """
        Deploy model for inference (replaces roboflow-python deploy)
        
        Args:
            model_path: Path to trained model
        """
        self.server = InferenceServer(self.config)
        return self.server.deploy(model_path)
    
    def track(self, video_path: str, model_path: str = None):
        """
        Track objects in video
        
        Args:
            video_path: Path to video file
            model_path: Path to detection model
        """
        self.tracker = ObjectTracker(self.config['tracking'])
        return self.tracker.track_video(video_path, model_path)