from typing import Generator, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
import uuid
import torch
import torchaudio

@dataclass
class AudioChunk:
    audio_path: str
    duration: float
    text: str

class BaseTTSModel:
    def generate(self, text: str) -> Generator[AudioChunk, None, None]:
        """Generate audio for text. Should be implemented by subclasses."""
        raise NotImplementedError

    def chunk_text(self, text: str) -> list[str]:
        """Break text into appropriate chunks for the model."""
        raise NotImplementedError

class KokoroTTS(BaseTTSModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.static_dir = Path('static/audio')
        self.static_dir.mkdir(exist_ok=True, parents=True)

    def chunk_text(self, text: str) -> list[str]:
        """Break text into sentences or smaller chunks if needed."""
        sentences = []
        current = ""
        
        # Split text into words while preserving spaces
        words = []
        current_word = ""
        for char in text:
            if char.isspace():
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            else:
                current_word += char
        if current_word:
            words.append(current_word)
        
        for word in words:
            # If adding this word would make the chunk too long
            if len(current) + len(word) > 200:
                # Only split if we have content
                if current:
                    sentences.append(current)
                    current = word
            else:
                current += word
            
            # If we hit a sentence boundary and have content, create a new chunk
            if word.strip() and word[-1] in '.!?' and current:
                sentences.append(current)
                current = ""
        
        # Don't forget the last chunk
        if current:
            sentences.append(current)
        
        return sentences

    def generate(self, text: str) -> Generator[AudioChunk, None, None]:
        """Generate audio chunks from text."""
        chunks = self.chunk_text(text)
        
        for chunk in chunks:
            if not chunk.strip():
                continue

            # Generate unique filename for this chunk
            filename = f"{uuid.uuid4()}.wav"
            filepath = self.static_dir / filename
            
            # Generate audio using Kokoro
            generator = self.pipeline(chunk, voice='af_heart', speed=1)
            _, _, audio = next(generator)
            
            # Save audio chunk
            import soundfile as sf
            sf.write(str(filepath), audio, 24000)
            
            # Calculate duration
            duration = len(audio) / 24000  # sampling rate is 24kHz
            
            yield AudioChunk(
                audio_path=f'/static/audio/{filename}',
                duration=duration,
                text=chunk
            )

class ZonosTTS(BaseTTSModel):
    def __init__(self, model_path: str, device: str = None):
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE
        
        self.device = device or DEFAULT_DEVICE
        self.model = Zonos.from_pretrained(model_path, device=self.device)
        # Load example audio for speaker embedding
        example_wav, example_sr = torchaudio.load("models/tts/zonos/example.mp3")
        self.speaker = self.model.make_speaker_embedding(example_wav, example_sr)
        self.static_dir = Path('static/audio')
        self.static_dir.mkdir(exist_ok=True, parents=True)

    def chunk_text(self, text: str) -> list[str]:
        """Break text into small chunks suitable for Zonos (max ~15 words per chunk)."""
        MAX_WORDS = 15
        chunks = []
        current = []
        word_count = 0
        
        # Split into sentences first
        sentences = re.split(r'([.!?]+)', text)
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            
            if not sentence:
                continue
                
            # Split sentence into words
            words = sentence.split()
            
            # Process each word
            for word in words:
                current.append(word)
                word_count += 1
                
                # If we hit the word limit or it's a natural break point
                if word_count >= MAX_WORDS:
                    # Add punctuation if it's the last part of the sentence
                    if not words[words.index(word) + 1:]:
                        current.append(punctuation)
                    else:
                        current.append(".")
                    
                    chunks.append(" ".join(current))
                    current = []
                    word_count = 0
            
            # Handle any remaining words in the sentence
            if current:
                current.append(punctuation)
                chunks.append(" ".join(current))
                current = []
                word_count = 0
        
        return chunks

    def generate(self, text: str) -> Generator[AudioChunk, None, None]:
        try:
            # Split text into manageable chunks
            chunks = self.chunk_text(text)
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                # Set random seed for consistent generation
                torch.manual_seed(421)
                
                # Generate unique filename for this chunk
                filename = f"{uuid.uuid4()}.wav"
                filepath = self.static_dir / filename
                
                # Prepare conditioning
                cond_dict = make_cond_dict(text=chunk, speaker=self.speaker, language="en-us")
                conditioning = self.model.prepare_conditioning(cond_dict)
                
                # Generate audio codes
                codes = self.model.generate(conditioning)
                
                # Decode to waveform
                wavs = self.model.autoencoder.decode(codes).cpu()
                
                # Save audio
                torchaudio.save(str(filepath), wavs[0], self.model.autoencoder.sampling_rate)
                
                # Calculate duration
                duration = wavs[0].shape[1] / self.model.autoencoder.sampling_rate
                
                yield AudioChunk(
                    audio_path=f'/static/audio/{filename}',
                    duration=duration,
                    text=chunk
                )
                
        except Exception as e:
            print(f"Error generating audio with Zonos: {e}")

def get_tts_model(model_type: str, **kwargs) -> Optional[BaseTTSModel]:
    """Factory function to create appropriate TTS model."""
    if model_type == 'kokoro':
        return KokoroTTS(kwargs['pipeline'])
    elif model_type == 'zonos':
        return ZonosTTS(
            model_path=kwargs.get('model_path', "Zyphra/Zonos-v0.1-transformer"),
            device=kwargs.get('device')
        )
    return None 