import os
import requests
from tqdm import tqdm
import torch
from pathlib import Path
import hashlib
from typing import Dict, Optional, Set
from threading import Lock

class ModelManager:
    MODELS = {
        # LLM Models
        'mistral-7b': {
            'name': 'Mistral 7B Instruct v0.1',
            'url': 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            'filename': 'mistral-7b-instruct-v0.1.Q4_K_M.gguf',
            'type': 'llm',
            'size': '4.37GB',
            'description': 'A powerful language model for chat and text generation',
            'sha256': None
        },
        
        # TTS Models
        'kokoro': {
            'name': 'Kokoro 82M',
            'files': {  # All required files for Kokoro
                'model': {
                    'url': 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth',
                    'path': 'kokoro-v1_0.pth'
                },
                'config': {
                    'url': 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json',
                    'path': 'config.json'
                },
                'voices/af_heart': {
                    'url': 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt',
                    'path': 'voices/af_heart.pt'
                }
            },
            'filename': 'kokoro-v1_0.pth',  # Main model file
            'type': 'tts',
            'subdir': 'kokoro',
            'size': '327MB',
            'description': 'Lightweight yet high-quality text-to-speech model (requires kokoro>=0.3.4 package)'
        },
        'zonos': {
            'name': 'Zonos v0.1 Transformer',
            'files': {
                'model': {
                    'url': 'https://huggingface.co/Zyphra/Zonos-v0.1-transformer/resolve/main/model.safetensors',
                    'path': 'model.safetensors'
                },
                'config': {
                    'url': 'https://huggingface.co/Zyphra/Zonos-v0.1-transformer/resolve/main/config.json',
                    'path': 'config.json'
                },
                'example': {
                    'url': 'https://huggingface.co/Zyphra/Zonos-v0.1-transformer/resolve/main/assets/exampleaudio.mp3',
                    'path': 'example.mp3'
                }
            },
            'filename': 'model.safetensors',
            'type': 'tts',
            'subdir': 'zonos',
            'size': '1.2GB',
            'description': 'High-quality text-to-speech model with voice cloning capabilities (requires espeak-ng and zonos package)'
        },
        
        # Speech Recognition Models
        'whisper-large': {
            'name': 'Whisper Large v3',
            'url': 'https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors',
            'filename': 'whisper-large-v3.safetensors',
            'type': 'stt',
            'size': '3.09GB',
            'description': 'OpenAI Whisper large model for speech recognition',
            'sha256': None
        },
        'whisper-small': {
            'name': 'Whisper Small',
            'url': 'https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors',
            'filename': 'whisper-small.safetensors',
            'type': 'stt',
            'size': '967MB',
            'description': 'OpenAI Whisper small model for speech recognition (244M params)',
            'sha256': None
        }
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different model types
        (self.models_dir / "llm").mkdir(exist_ok=True)
        (self.models_dir / "tts").mkdir(exist_ok=True)
        (self.models_dir / "stt").mkdir(exist_ok=True)

        # Track downloads in progress
        self.downloads_in_progress: Set[str] = set()
        self.download_lock = Lock()

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get the path to a downloaded model, return None if not downloaded."""
        if model_id not in self.MODELS:
            return None
            
        model_info = self.MODELS[model_id]
        if 'files' in model_info:
            # For models with multiple files (like Kokoro)
            model_dir = self.models_dir / model_info['type'] / model_info['subdir']
            # Check if all required files exist
            for file_info in model_info['files'].values():
                file_path = model_dir / file_info['path']
                if not file_path.exists():
                    return None
            # Return path to main model file
            return model_dir / model_info['filename']
        else:
            model_path = self.models_dir / model_info['type'] / model_info['filename']
            return model_path if model_path.exists() else None

    def is_downloading(self, model_id: str) -> bool:
        """Check if a model is currently being downloaded."""
        with self.download_lock:
            return model_id in self.downloads_in_progress

    def download_model(self, model_id: str, progress_callback=None) -> tuple[bool, str]:
        """Download a model and return (success, error_message)."""
        if model_id not in self.MODELS:
            return False, "Model not found"
            
        # Check if already downloading
        with self.download_lock:
            if model_id in self.downloads_in_progress:
                return False, "Download already in progress"
            self.downloads_in_progress.add(model_id)
            # Return immediately to show downloading status
            return True, ""

    def _download_model_async(self, model_id: str, progress_callback=None):
        """Actually perform the download in background."""
        try:
            model_info = self.MODELS[model_id]
            
            if 'files' in model_info:  # For models with multiple files (like Kokoro)
                model_dir = self.models_dir / model_info['type'] / model_info['subdir']
                model_dir.mkdir(exist_ok=True, parents=True)
                
                # Download all required files
                for file_info in model_info['files'].values():
                    file_path = model_dir / file_info['path']
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    if not file_path.exists():
                        self._download_file(file_info['url'], file_path, progress_callback)
                
            else:  # For simple model files
                save_path = self.models_dir / model_info['type'] / model_info['filename']
                if not save_path.exists():
                    self._download_file(model_info['url'], save_path, progress_callback)
            
        except Exception as e:
            # Clean up any partially downloaded files
            if 'files' in model_info:
                model_dir = self.models_dir / model_info['type'] / model_info['subdir']
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
            print(f"Download failed: {str(e)}")
        finally:
            # Remove from downloads in progress
            with self.download_lock:
                self.downloads_in_progress.remove(model_id)

    def start_download(self, model_id: str, progress_callback=None):
        """Start the download process."""
        import threading
        success, error = self.download_model(model_id, progress_callback)
        if success:
            thread = threading.Thread(
                target=self._download_model_async,
                args=(model_id, progress_callback)
            )
            thread.start()
        return success, error

    def _download_file(self, url: str, path: Path, progress_callback=None):
        """Helper function to download a file with progress tracking."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    if progress_callback:
                        progress = (pbar.n / total_size) * 100
                        progress_callback(progress)

    def list_available_models(self) -> Dict[str, dict]:
        """Return detailed information about all models."""
        return {
            model_id: {
                **model_info,
                'installed': self.get_model_path(model_id) is not None,
                'downloading': self.is_downloading(model_id)
            }
            for model_id, model_info in self.MODELS.items()
        }

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a specific model."""
        return self.MODELS.get(model_id) 