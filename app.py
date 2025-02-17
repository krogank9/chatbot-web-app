from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
from llama_cpp import Llama
import torch
import sys
import os
from model_manager import ModelManager
import scipy.io.wavfile as wavfile
import psutil
from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from chat_manager import ChatManager
import json
import time
import soundfile as sf
from tts_manager import get_tts_model, AudioChunk
import tempfile
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
model_manager = ModelManager()
chat_manager = ChatManager()

# Add TTS model paths to Python path
tts_path = Path("models/tts")
if tts_path.exists():
    for tts_model_dir in tts_path.iterdir():
        if tts_model_dir.is_dir():
            sys.path.append(str(tts_model_dir.absolute()))

@dataclass
class LoadedModels:
    llm: Optional[str] = None
    tts: Optional[str] = None
    stt: Optional[str] = None

# Initialize models to None
llm = None
kokoro_pipeline = None

# Add after model_manager initialization
loaded_models = LoadedModels()

# Add Whisper model variables
whisper_model = None
whisper_processor = None

def initialize_models():
    global llm, kokoro_pipeline, whisper_model, whisper_processor, loaded_models
    
    # Initialize Mistral if available
    mistral_path = model_manager.get_model_path('mistral-7b')
    if mistral_path:
        try:
            # Don't load immediately, just mark as available
            loaded_models.llm = None
        except Exception as e:
            print(f"Failed to load Mistral: {e}")

    # Initialize Kokoro if available
    kokoro_path = model_manager.get_model_path('kokoro')
    if kokoro_path:
        try:
            try:
                from kokoro import KPipeline
            except ImportError:
                print("Kokoro package not found. Install with: pip install kokoro>=0.3.4")
                return
                
            # Initialize Kokoro pipeline
            kokoro_pipeline = KPipeline(lang_code='a')  # 'a' for American English
            loaded_models.tts = 'kokoro'
        except Exception as e:
            print(f"Failed to load Kokoro: {e}")
            
    # Initialize Whisper
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        ).to(device)
        loaded_models.stt = 'whisper-small'
        print("Whisper model loaded successfully")
    except Exception as e:
        print(f"Failed to load Whisper: {e}")

def generate_audio(text: str) -> Optional[str]:
    """Generate audio from text using Kokoro TTS."""
    if not kokoro_pipeline:
        return None
        
    try:
        # Generate audio using pipeline
        generator = kokoro_pipeline(text, voice='af_heart', speed=1)
        # Get first (and only) chunk
        _, _, audio = next(generator)
        # Save audio
        audio_path = Path('static/response.wav')
        sf.write(str(audio_path), audio, 24000)
        return '/static/response.wav'
    except Exception as e:
        print(f"Failed to generate audio: {e}")
        return None

initialize_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/models/status')
def get_models_status():
    return jsonify(model_manager.list_available_models())

@app.route('/models/status/stream')
def stream_model_status():
    def generate():
        while True:
            models = model_manager.list_available_models()
            yield f"data: {json.dumps(models)}\n\n"
            time.sleep(0.5)  # Update every 500ms
            
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/models/download', methods=['POST'])
def download_model():
    model_id = request.json['model_id']
    success, error = model_manager.start_download(model_id)
    return jsonify({
        'success': success,
        'error': error
    })

@app.route('/models/load', methods=['POST'])
def load_model():
    global llm, kokoro_pipeline, whisper_model, whisper_processor, loaded_models
    
    model_id = request.json['model_id']
    model_info = model_manager.get_model_info(model_id)
    
    if not model_info:
        return jsonify({'success': False, 'error': 'Model not found'})
        
    try:
        if model_info['type'] == 'llm':
            # Unload previous LLM if exists
            if llm:
                del llm
                
            model_path = model_manager.get_model_path(model_id)
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,
                n_ctx=2048,
                n_batch=512,
                verbose=False,
                offload_kqv=True,
                use_mlock=True,
                use_mmap=True,
            )
            loaded_models.llm = model_id
            
        elif model_info['type'] == 'tts':
            # Unload previous TTS if exists
            if kokoro_pipeline:
                del kokoro_pipeline
                kokoro_pipeline = None
            
            model_path = model_manager.get_model_path(model_id)
            if not model_path:
                return jsonify({'success': False, 'error': f'{model_id} model files not found'})
            
            if model_id == 'kokoro':
                try:
                    from kokoro import KPipeline
                except ImportError:
                    return jsonify({'success': False, 'error': 'Kokoro package not found. Install with: pip install kokoro>=0.3.4'})
                    
                kokoro_pipeline = KPipeline(lang_code='a')  # 'a' for American English
                loaded_models.tts = model_id
            elif model_id == 'zonos':
                try:
                    import zonos
                except ImportError:
                    return jsonify({'success': False, 'error': 'Zonos package not found. Install with: pip install -e .'})
                
                try:
                    import espeak
                except ImportError:
                    return jsonify({'success': False, 'error': 'eSpeak not found. Install with: apt install -y espeak-ng'})
                
                # Zonos will be loaded on-demand in the TTS manager
                loaded_models.tts = model_id
            
        elif model_info['type'] == 'stt':
            # Unload previous STT if exists
            if whisper_model:
                del whisper_model
                del whisper_processor
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-small"
            ).to(device)
            loaded_models.stt = model_id
            
        return jsonify({
            'success': True,
            'loaded_models': {
                'llm': loaded_models.llm,
                'tts': loaded_models.tts,
                'stt': loaded_models.stt
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/models/loaded')
def get_loaded_models():
    return jsonify({
        'llm': loaded_models.llm,
        'tts': loaded_models.tts,
        'stt': loaded_models.stt
    })

@app.route('/chat', methods=['POST'])
def chat():
    global llm, kokoro_pipeline
    
    chat_id = request.json.get('chat_id')
    if not chat_id:
        return jsonify({'error': 'No chat ID provided'}), 400
        
    chat = chat_manager.load_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    persona = chat_manager.get_persona(chat.persona_id)
    if not persona:
        return jsonify({'error': 'Persona not found'}), 404

    # Load model if not loaded
    if not llm and model_manager.get_model_path('mistral-7b'):
        try:
            mistral_path = model_manager.get_model_path('mistral-7b')
            llm = Llama(
                model_path=str(mistral_path),
                n_gpu_layers=-1,
                n_ctx=2048,
                n_batch=512,
                verbose=False,
                offload_kqv=True,
                use_mlock=True,
                use_mmap=True,
            )
            loaded_models.llm = 'mistral-7b'
        except Exception as e:
            return jsonify({'error': f'Failed to load LLM: {str(e)}'}), 500
    elif not llm:
        return jsonify({'error': 'No LLM model available'}), 400
        
    user_message = request.json['message']
    chat_manager.add_message(chat_id, 'user', user_message)
    
    messages = [msg.content for msg in chat.messages]
    prompt = f"""<s>[INST] {persona.system_prompt}

Current conversation:
{' '.join(messages)}
Human: {user_message}
Assistant: [/INST]"""

    # Get TTS model if available
    tts_model = None
    if kokoro_pipeline and loaded_models.tts == 'kokoro':
        tts_model = get_tts_model('kokoro', pipeline=kokoro_pipeline)
    
    def generate():
        full_response = ""
        current_chunk = ""
        
        for chunk in llm(
            prompt,
            max_tokens=150,
            temperature=0.8,
            stop=["</s>", "[INST]", "Human:"],
            echo=False,
            stream=True
        ):
            if chunk['choices'][0]['text']:
                text = chunk['choices'][0]['text']
                full_response += text
                current_chunk += text
                
                # Check if we have a complete sentence or reached chunk size limit
                if text.endswith(('.', '!', '?', '\n')) or len(current_chunk) > 200:
                    if tts_model:
                        try:
                            for audio_chunk in tts_model.generate(current_chunk):
                                yield f"data: {json.dumps({'text': text, 'done': False, 'audio': audio_chunk.__dict__})}\n\n"
                        except Exception as e:
                            print(f"TTS generation failed: {e}")
                    else:
                        yield f"data: {json.dumps({'text': text, 'done': False})}\n\n"
                    current_chunk = ""
                else:
                    yield f"data: {json.dumps({'text': text, 'done': False})}\n\n"
        
        # Save the complete message
        chat_manager.add_message(chat_id, 'assistant', full_response)
        
        # Send final chunk
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/system/stats')
def get_system_stats():
    # CPU stats
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_freq = psutil.cpu_freq()
    cpu_count = psutil.cpu_count()
    
    # Memory stats
    memory = psutil.virtual_memory()
    
    # GPU stats
    gpu_stats = []
    try:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_free': gpu.memoryFree,
                    'gpu_load': gpu.load * 100,
                    'temperature': gpu.temperature
                })
        except ImportError:
            pass  # GPUtil not installed
    except:
        pass  # No GPU or GPUtil not working
        
    return jsonify({
        'cpu': {
            'usage_percent': cpu_percent,
            'frequency': round(cpu_freq.current / 1000, 2) if cpu_freq else None,  # GHz
            'cores': cpu_count
        },
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        },
        'gpu': gpu_stats
    })

@app.route('/personas', methods=['GET'])
def list_personas():
    return jsonify([
        {
            'id': p.id,
            'name': p.name,
            'description': p.description,
            'created_at': p.created_at
        }
        for p in chat_manager.personas.values()
    ])

@app.route('/personas', methods=['POST'])
def create_persona():
    data = request.json
    try:
        persona = chat_manager.create_persona(
            name=data['name'],
            system_prompt=data['system_prompt'],
            greeting=data['greeting'],
            description=data['description']
        )
        return jsonify({
            'id': persona.id,
            'name': persona.name,
            'description': persona.description,
            'created_at': persona.created_at
        })
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400

@app.route('/chats', methods=['GET'])
def list_chats():
    return jsonify(chat_manager.list_chats())

@app.route('/chats', methods=['POST'])
def create_chat():
    persona_id = request.json.get('persona_id')
    if not persona_id:
        return jsonify({'error': 'No persona ID provided'}), 400
        
    try:
        chat = chat_manager.create_chat(persona_id)
        return jsonify({
            'id': chat.id,
            'persona_id': chat.persona_id,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'audio_url': msg.audio_url
                }
                for msg in chat.messages
            ],
            'created_at': chat.created_at,
            'updated_at': chat.updated_at
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    chat = chat_manager.load_chat(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    return jsonify({
        'id': chat.id,
        'persona_id': chat.persona_id,
        'messages': [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'audio_url': msg.audio_url
            }
            for msg in chat.messages
        ],
        'created_at': chat.created_at,
        'updated_at': chat.updated_at
    })

@app.route('/personas/manage')
def manage_personas():
    return render_template('personas.html')

@app.route('/personas/<persona_id>', methods=['GET'])
def get_persona(persona_id):
    persona = chat_manager.get_persona(persona_id)
    if not persona:
        return jsonify({'error': 'Persona not found'}), 404
    return jsonify({
        'id': persona.id,
        'name': persona.name,
        'description': persona.description,
        'system_prompt': persona.system_prompt,
        'greeting': persona.greeting,
        'created_at': persona.created_at
    })

@app.route('/personas/<persona_id>', methods=['PUT'])
def update_persona(persona_id):
    data = request.json
    try:
        persona = chat_manager.update_persona(
            persona_id,
            name=data['name'],
            system_prompt=data['system_prompt'],
            greeting=data['greeting'],
            description=data['description']
        )
        return jsonify({
            'id': persona.id,
            'name': persona.name,
            'description': persona.description,
            'created_at': persona.created_at
        })
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    try:
        chat_manager.delete_chat(chat_id)
        return '', 204
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@app.route('/generate_audio', methods=['POST'])
def generate_final_audio():
    """Generate audio for any remaining text."""
    text = request.json.get('text')
    if not text or not kokoro_pipeline:
        return jsonify({'success': False})
        
    try:
        tts_model = get_tts_model('kokoro', pipeline=kokoro_pipeline)
        for audio_chunk in tts_model.generate(text):
            # Just generate the audio, frontend will handle it
            pass
        return jsonify({'success': True})
    except Exception as e:
        print(f"Failed to generate final audio: {e}")
        return jsonify({'success': False})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if not whisper_model or not whisper_processor:
        return jsonify({'error': 'Speech-to-text model not loaded'}), 400

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            
            try:
                # Load and preprocess the audio using scipy
                sr, audio = wavfile.read(temp_file.name)
                
                # Convert to float32 and normalize
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                
                # Ensure mono audio
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                
                # Resample to 16kHz if needed
                if sr != 16000:
                    import resampy
                    audio = resampy.resample(audio, sr, 16000)
                    sr = 16000
                
                # Process with Whisper
                input_features = whisper_processor(
                    audio, 
                    sampling_rate=sr, 
                    return_tensors="pt"
                ).input_features.to(whisper_model.device)
                
                # Generate token ids
                predicted_ids = whisper_model.generate(input_features)
                
                # Decode token ids to text
                transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
                text = transcription[0].strip()
                
                return jsonify({'text': text})
            except Exception as e:
                print(f"Transcription error details: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full traceback for debugging
                return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
            finally:
                # Clean up the temporary file
                os.unlink(temp_file.name)
    except Exception as e:
        print(f"Audio processing error details: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({'error': f'Failed to process audio: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)