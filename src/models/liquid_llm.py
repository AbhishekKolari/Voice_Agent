"""
Custom LangChain LLM wrapper for Liquid AI LFM2-Audio-1.5B model
"""
import torch
import torchaudio
from typing import Optional, List, Any
from dataclasses import dataclass
from pathlib import Path

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
from langchain.llms.base import LLM
from pydantic import Field

from config import Config


@dataclass
class AudioResponse:
    """Container for audio generation outputs"""
    text: str
    audio_tokens: List[torch.Tensor]
    waveform: Optional[torch.Tensor] = None
    audio_path: Optional[str] = None


class LiquidAudioLLM(LLM):
    """
    Custom LangChain wrapper for Liquid AI LFM2-Audio-1.5B model
    
    Supports three generation modes:
    1. Text-to-text (interleaved): For agent reasoning and multi-turn chat
    2. Audio-to-text (sequential): For speech-to-text transcription
    3. Text-to-audio (sequential): For text-to-speech generation
    """
    
    model: Any = Field(default=None, exclude=True)
    processor: Any = Field(default=None, exclude=True)
    chat_state: Any = Field(default=None, exclude=True)
    device: str = Config.DEVICE
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the Liquid AI model and processor"""
        print(f"Loading Liquid AI model from {Config.HF_REPO}...")
        print(f"Using device: {self.device}")
        
        self.processor = LFM2AudioProcessor.from_pretrained(Config.HF_REPO).eval()
        self.model = LFM2AudioModel.from_pretrained(Config.HF_REPO).eval()
        
        # Only move to CUDA if explicitly using cuda AND it's available
        if self.device == "cuda":
            try:
                self.model = self.model.cuda()
                print("Model successfully loaded on GPU")
            except Exception as e:
                print(f"Warning: Failed to load model on GPU: {e}")
                print("Falling back to CPU")
                self.device = "cpu"
        else:
            print("Model loaded on CPU")
        
        # Initialize chat state for multi-turn conversations
        self.chat_state = ChatState(self.processor)
        self._setup_system_prompt()
        
        print("Model loaded successfully!")
    
    def _setup_system_prompt(self, prompt: str = Config.SYSTEM_PROMPT):
        """Set up initial system prompt"""
        self.chat_state.new_turn("system")
        self.chat_state.add_text(prompt)
        self.chat_state.end_turn()
    
    @property
    def _llm_type(self) -> str:
        return "liquid_audio"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs
    ) -> str:
        """
        Main inference method for LangChain integration (multi-turn chat)
        Uses generate_interleaved for conversational context
        
        Args:
            prompt: Text input from user
            
        Returns:
            Text response from model
        """
        # Add user message to chat
        self.chat_state.new_turn("user")
        self.chat_state.add_text(prompt)
        self.chat_state.end_turn()
        
        # Generate response
        self.chat_state.new_turn("assistant")
        
        text_out = []
        audio_out = []
        modality_out = []
        
        # Use generate_interleaved for multi-turn conversation
        # Note: generate_interleaved only supports audio_temperature and audio_top_k
        with torch.no_grad():
            for t in self.model.generate_interleaved(
                **self.chat_state,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                audio_temperature=Config.AUDIO_TEMPERATURE,
                audio_top_k=Config.AUDIO_TOP_K
            ):
                if t.numel() == 1:
                    text_out.append(t)
                    modality_out.append(LFMModality.TEXT)
                else:
                    audio_out.append(t)
                    modality_out.append(LFMModality.AUDIO_OUT)
        
        # Convert tokens to text
        response_text = "".join([
            self.processor.text.decode(t) for t in text_out
        ])
        
        # Update chat history
        if text_out:
            self.chat_state.append(
                text=torch.stack(text_out, 1) if len(text_out) > 1 else text_out[0].unsqueeze(1),
                audio_out=torch.stack(audio_out, 1) if audio_out else None,
                modality_flag=torch.tensor(modality_out)
            )
        
        self.chat_state.end_turn()
        
        return response_text
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text (Speech-to-Text)
        Uses generate_sequential for single-turn transcription
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Load audio
        wav, sampling_rate = torchaudio.load(audio_path)
        
        # Create a fresh chat state for sequential generation
        temp_chat = ChatState(self.processor)
        
        # Set up dictation system prompt
        temp_chat.new_turn("system")
        temp_chat.add_text(Config.DICTATION_SYSTEM_PROMPT)
        temp_chat.end_turn()
        
        # Add audio input
        temp_chat.new_turn("user")
        temp_chat.add_audio(wav, sampling_rate)
        temp_chat.end_turn()
        
        temp_chat.new_turn("assistant")
        
        # Use generate_sequential for single audio-to-text task
        # generate_sequential also uses audio_temperature and audio_top_k
        text_out = []
        with torch.no_grad():
            for t in self.model.generate_sequential(
                **temp_chat,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                audio_temperature=Config.AUDIO_TEMPERATURE,
                audio_top_k=Config.AUDIO_TOP_K
            ):
                if t.numel() == 1:
                    text_out.append(t)
        
        transcription = "".join([self.processor.text.decode(t) for t in text_out])
        
        return transcription.strip()
    
    def generate_audio_response(
        self, 
        prompt: str, 
        output_path: Optional[str] = None
    ) -> AudioResponse:
        """
        Generate audio response from text (Text-to-Speech)
        Uses generate_sequential for single-turn TTS
        
        Args:
            prompt: Text prompt to convert to speech
            output_path: Path to save audio file (auto-generated if None)
            
        Returns:
            AudioResponse containing text, audio tokens, and waveform
        """
        # Create a fresh chat state for sequential generation
        temp_chat = ChatState(self.processor)
        
        # Set up TTS system prompt
        temp_chat.new_turn("system")
        temp_chat.add_text("Generate natural speech audio for the following text.")
        temp_chat.end_turn()
        
        # Add text input
        temp_chat.new_turn("user")
        temp_chat.add_text(prompt)
        temp_chat.end_turn()
        
        temp_chat.new_turn("assistant")
        
        text_out = []
        audio_out = []
        
        # Use generate_sequential for single text-to-audio task
        with torch.no_grad():
            for t in self.model.generate_sequential(
                **temp_chat,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                audio_temperature=Config.AUDIO_TEMPERATURE,
                audio_top_k=Config.AUDIO_TOP_K
            ):
                if t.numel() == 1:
                    text_out.append(t)
                else:
                    audio_out.append(t)
        
        response_text = "".join([self.processor.text.decode(t) for t in text_out])
        
        # Decode audio if generated
        waveform = None
        saved_path = None
        if audio_out and len(audio_out) > 1:
            # Remove the last "end-of-audio" token
            mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            with torch.no_grad():
                waveform = self.processor.mimi.decode(mimi_codes)[0]
            
            # Save audio
            if output_path is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(Config.AUDIO_OUTPUT_DIR / f"tts_{timestamp}.wav")
            
            torchaudio.save(output_path, waveform.cpu(), Config.SAMPLE_RATE)
            saved_path = output_path
            print(f"Audio saved to: {output_path}")
        
        return AudioResponse(
            text=response_text,
            audio_tokens=audio_out,
            waveform=waveform,
            audio_path=saved_path
        )
    
    def chat_with_audio_response(
        self, 
        prompt: str, 
        output_path: Optional[str] = None
    ) -> AudioResponse:
        """
        Multi-turn chat with audio response
        Uses generate_interleaved to maintain conversation context
        
        Args:
            prompt: Text input from user
            output_path: Path to save audio file
            
        Returns:
            AudioResponse with text and audio
        """
        # Add user message to chat state
        self.chat_state.new_turn("user")
        self.chat_state.add_text(prompt)
        self.chat_state.end_turn()
        
        self.chat_state.new_turn("assistant")
        
        text_out = []
        audio_out = []
        modality_out = []
        
        # Use generate_interleaved for multi-turn with audio
        with torch.no_grad():
            for t in self.model.generate_interleaved(
                **self.chat_state,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                audio_temperature=Config.AUDIO_TEMPERATURE,
                audio_top_k=Config.AUDIO_TOP_K
            ):
                if t.numel() == 1:
                    text_out.append(t)
                    modality_out.append(LFMModality.TEXT)
                else:
                    audio_out.append(t)
                    modality_out.append(LFMModality.AUDIO_OUT)
        
        response_text = "".join([self.processor.text.decode(t) for t in text_out])
        
        # Decode audio
        waveform = None
        saved_path = None
        if audio_out and len(audio_out) > 1:
            mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
            with torch.no_grad():
                waveform = self.processor.mimi.decode(mimi_codes)[0]
            
            if output_path is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = str(Config.AUDIO_OUTPUT_DIR / f"chat_{timestamp}.wav")
            
            torchaudio.save(output_path, waveform.cpu(), Config.SAMPLE_RATE)
            saved_path = output_path
        
        # Update chat history
        if text_out:
            self.chat_state.append(
                text=torch.stack(text_out, 1) if len(text_out) > 1 else text_out[0].unsqueeze(1),
                audio_out=torch.stack(audio_out, 1) if audio_out else None,
                modality_flag=torch.tensor(modality_out)
            )
        
        self.chat_state.end_turn()
        
        return AudioResponse(
            text=response_text,
            audio_tokens=audio_out,
            waveform=waveform,
            audio_path=saved_path
        )
    
    def reset_conversation(self):
        """Clear chat history and reinitialize"""
        self.chat_state = ChatState(self.processor)
        self._setup_system_prompt()
        print("Conversation reset.")