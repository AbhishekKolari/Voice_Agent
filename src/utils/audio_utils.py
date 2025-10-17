"""
Audio utilities for recording and playback
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import Config


class AudioRecorder:
    """
    Record audio from microphone
    """
    
    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = []
        self.is_recording = False
        
    def start_recording(self):
        """Start recording audio"""
        self.recording = []
        self.is_recording = True
        print("🎤 Recording started... (Press Enter to stop)")
        
    def record_chunk(self, duration: float = 0.1):
        """Record a chunk of audio"""
        if self.is_recording:
            chunk = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            self.recording.append(chunk)
    
    def stop_recording(self, output_path: Optional[str] = None) -> str:
        """Stop recording and save to file"""
        self.is_recording = False
        
        if not self.recording:
            print("No audio recorded")
            return None
        
        # Concatenate all chunks
        audio_data = np.concatenate(self.recording, axis=0)
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Config.AUDIO_OUTPUT_DIR / f"recording_{timestamp}.wav")
        
        # Save audio file
        sf.write(output_path, audio_data, self.sample_rate)
        print(f"✓ Recording saved to: {output_path}")
        
        return output_path
    
    def record_fixed_duration(self, duration: float, output_path: Optional[str] = None) -> str:
        """Record audio for a fixed duration"""
        print(f"🎤 Recording for {duration} seconds...")
        
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Config.AUDIO_OUTPUT_DIR / f"recording_{timestamp}.wav")
        
        # Save audio file
        sf.write(output_path, audio_data, self.sample_rate)
        print(f"✓ Recording saved to: {output_path}")
        
        return output_path


class AudioPlayer:
    """
    Play audio files
    """
    
    @staticmethod
    def play(audio_path: str):
        """Play an audio file"""
        try:
            data, sample_rate = sf.read(audio_path)
            print(f"🔊 Playing: {audio_path}")
            sd.play(data, sample_rate)
            sd.wait()
            print("✓ Playback complete")
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    @staticmethod
    def play_async(audio_path: str):
        """Play an audio file asynchronously (non-blocking)"""
        try:
            data, sample_rate = sf.read(audio_path)
            print(f"🔊 Playing: {audio_path}")
            sd.play(data, sample_rate)
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    @staticmethod
    def stop():
        """Stop any currently playing audio"""
        sd.stop()