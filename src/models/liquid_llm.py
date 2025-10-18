"""
Custom LangChain LLM wrapper for Liquid AI LFM2-Audio-1.5B model
Following official Liquid AI patterns from their GitHub repo
"""
import torch
import torchaudio
from typing import Optional, List, Any
from dataclasses import dataclass
from pathlib import Path
import re

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
    
    Supports three generation modes (following official Liquid AI patterns):
    1. Multi-turn multimodal chat (generate_interleaved): Audio/Text input → Text+Audio output
    2. Pure ASR (generate_sequential): Audio input → Text output
    3. Pure TTS (generate_sequential): Text input → Audio output
    """
    
    model: Any = Field(default=None, exclude=True)
    processor: Any = Field(default=None, exclude=True)
    chat_state: Any = Field(default=None, exclude=True)
    device: str = Config.DEVICE
    conversation_turns: int = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the Liquid AI model and processor"""
        print(f"Loading Liquid AI model from {Config.HF_REPO}...")
        print(f"Using device: {self.device}")
        
        self.processor = LFM2AudioProcessor.from_pretrained(Config.HF_REPO).eval()
        self.model = LFM2AudioModel.from_pretrained(Config.HF_REPO).eval()
        
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
    
    def _setup_system_prompt(self, prompt: str = "Respond with interleaved text and audio."):
        """Set up initial system prompt for multi-turn chat"""
        self.chat_state.new_turn("system")
        self.chat_state.add_text(prompt)
        self.chat_state.end_turn()
        self.conversation_turns = 0
    
    def _clean_response(self, text: str) -> str:
        """Clean up model artifacts from response"""
        # Remove special tokens and artifacts
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()
        return text
    
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
        """
        # Reset if too many turns
        if self.conversation_turns > 10:
            print("[Info] Resetting chat state after 10 turns...")
            self.reset_conversation()
        
        # Add user message
        self.chat_state.new_turn("user")
        self.chat_state.add_text(prompt)
        self.chat_state.end_turn()
        
        self.chat_state.new_turn("assistant")
        
        text_out = []
        audio_out = []
        modality_out = []
        
        try:
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
            response_text = self._clean_response(response_text)
            
            # Append to chat history
            if text_out:
                self.chat_state.append(
                    text=torch.stack(text_out, 1) if len(text_out) > 1 else text_out[0].unsqueeze(1),
                    audio_out=torch.stack(audio_out, 1) if audio_out else None,
                    modality_flag=torch.tensor(modality_out)
                )
            
            self.chat_state.end_turn()
            self.conversation_turns += 1
            
            return response_text
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e).lower():
                print(f"[Error] Generation error: {e}")
                print("[Action] Resetting conversation...")
                self.reset_conversation()
                return "I encountered an error. Let's start fresh. What would you like to talk about?"
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Pure ASR (Automatic Speech Recognition)
        Following official pattern: generate_sequential with "Perform ASR." system prompt
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio
            wav, sampling_rate = torchaudio.load(audio_path)
            
            # Create fresh chat state for ASR (following official pattern)
            chat = ChatState(self.processor)
            
            chat.new_turn("system")
            chat.add_text("Perform ASR.")  # Official ASR system prompt
            chat.end_turn()
            
            chat.new_turn("user")
            chat.add_audio(wav, sampling_rate)
            chat.end_turn()
            
            chat.new_turn("assistant")
            
            # Generate text only (following official pattern)
            text_out = []
            with torch.no_grad():
                for t in self.model.generate_sequential(
                    **chat,
                    max_new_tokens=Config.MAX_NEW_TOKENS
                ):
                    if t.numel() == 1:
                        text_out.append(t)
            
            transcription = "".join([self.processor.text.decode(t) for t in text_out])
            transcription = self._clean_response(transcription)
            
            return transcription
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise
    
    def generate_audio_response(
        self, 
        text: str, 
        output_path: Optional[str] = None,
        voice_description: str = None
    ) -> AudioResponse:
        """
        Pure TTS (Text-to-Speech)
        Following official pattern: generate_sequential with voice description
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice_description: Optional voice characteristics description
            
        Returns:
            AudioResponse containing audio
        """
        try:
            # Create fresh chat state for TTS (following official pattern)
            chat = ChatState(self.processor)
            
            chat.new_turn("system")
            # Official TTS system prompt with voice description
            if voice_description is None:
                voice_description = (
                    "A clear speaker delivers the text with a natural tone. "
                    "The recording is of good quality with minimal noise."
                )
            
            system_prompt = f"Perform TTS.\nUse the following voice: {voice_description}"
            chat.add_text(system_prompt)
            chat.end_turn()
            
            chat.new_turn("user")
            chat.add_text(text)
            chat.end_turn()
            
            chat.new_turn("assistant")
            
            # Generate audio only (following official pattern)
            audio_out = []
            text_out = []
            with torch.no_grad():
                for t in self.model.generate_sequential(
                    **chat,
                    max_new_tokens=Config.MAX_NEW_TOKENS,
                    audio_temperature=Config.TTS_AUDIO_TEMPERATURE,
                    audio_top_k=Config.TTS_AUDIO_TOP_K
                ):
                    if t.numel() > 1:
                        audio_out.append(t)
                    else:
                        # Some models might output text tokens too
                        text_out.append(t)
            
            response_text = ""
            if text_out:
                response_text = "".join([self.processor.text.decode(t) for t in text_out])
                response_text = self._clean_response(response_text)
            
            # Decode audio
            waveform = None
            saved_path = None
            
            if audio_out and len(audio_out) > 1:
                try:
                    # Remove last "end-of-audio" token
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
                    
                except Exception as e:
                    print(f"Warning: Could not decode/save audio: {e}")
            else:
                print("[Warning] No audio tokens generated. TTS may not have worked.")
            
            return AudioResponse(
                text=response_text or text,  # Return input text if no text output
                audio_tokens=audio_out,
                waveform=waveform,
                audio_path=saved_path
            )
            
        except Exception as e:
            print(f"Error during TTS: {e}")
            raise
    
    def chat_with_audio_response(
        self, 
        prompt: str, 
        output_path: Optional[str] = None
    ) -> AudioResponse:
        """
        Multi-turn chat with audio response
        Uses generate_interleaved (following official multi-turn pattern)
        
        Args:
            prompt: Text input from user
            output_path: Path to save audio file
            
        Returns:
            AudioResponse with text and audio
        """
        try:
            # Reset if too many turns
            if self.conversation_turns > 10:
                print("[Info] Resetting chat state after 10 turns...")
                self.reset_conversation()
            
            # Add user message
            self.chat_state.new_turn("user")
            self.chat_state.add_text(prompt)
            self.chat_state.end_turn()
            
            self.chat_state.new_turn("assistant")
            
            # Following official multi-turn pattern
            text_out = []
            audio_out = []
            modality_out = []
            
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
            response_text = self._clean_response(response_text)
            
            # Decode audio if present
            waveform = None
            saved_path = None
            if audio_out and len(audio_out) > 1:
                try:
                    mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
                    with torch.no_grad():
                        waveform = self.processor.mimi.decode(mimi_codes)[0]
                    
                    if output_path is None:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = str(Config.AUDIO_OUTPUT_DIR / f"chat_{timestamp}.wav")
                    
                    torchaudio.save(output_path, waveform.cpu(), Config.SAMPLE_RATE)
                    saved_path = output_path
                    print(f"Audio saved to: {output_path}")
                except Exception as e:
                    print(f"Warning: Could not decode/save audio: {e}")
            
            # Append to chat history (following official pattern)
            if text_out:
                self.chat_state.append(
                    text=torch.stack(text_out, 1) if len(text_out) > 1 else text_out[0].unsqueeze(1),
                    audio_out=torch.stack(audio_out, 1) if audio_out else None,
                    modality_flag=torch.tensor(modality_out)
                )
            
            self.chat_state.end_turn()
            self.conversation_turns += 1
            
            return AudioResponse(
                text=response_text,
                audio_tokens=audio_out,
                waveform=waveform,
                audio_path=saved_path
            )
            
        except RuntimeError as e:
            if "CUDA" in str(e) or "assert" in str(e).lower():
                print(f"[Error] Generation error: {e}")
                print("[Action] Resetting conversation...")
                self.reset_conversation()
                return AudioResponse(
                    text="I encountered an error. Let's start fresh.",
                    audio_tokens=[],
                    waveform=None,
                    audio_path=None
                )
            raise
    
    def process_audio_with_response(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> AudioResponse:
        """
        Process audio input and generate audio response
        Uses generate_interleaved with audio input (following official multi-turn pattern)
        
        Args:
            audio_path: Path to input audio
            output_path: Path to save response audio
            
        Returns:
            AudioResponse with transcription and audio response
        """
        try:
            # Load audio
            wav, sampling_rate = torchaudio.load(audio_path)
            
            # Reset if too many turns
            if self.conversation_turns > 10:
                print("[Info] Resetting chat state after 10 turns...")
                self.reset_conversation()
            
            # Add audio input (following official pattern)
            self.chat_state.new_turn("user")
            self.chat_state.add_audio(wav, sampling_rate)
            self.chat_state.end_turn()
            
            self.chat_state.new_turn("assistant")
            
            # Generate response
            text_out = []
            audio_out = []
            modality_out = []
            
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
            response_text = self._clean_response(response_text)
            
            # Decode audio
            waveform = None
            saved_path = None
            if audio_out and len(audio_out) > 1:
                try:
                    mimi_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)
                    with torch.no_grad():
                        waveform = self.processor.mimi.decode(mimi_codes)[0]
                    
                    if output_path is None:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = str(Config.AUDIO_OUTPUT_DIR / f"response_{timestamp}.wav")
                    
                    torchaudio.save(output_path, waveform.cpu(), Config.SAMPLE_RATE)
                    saved_path = output_path
                    print(f"Audio response saved to: {output_path}")
                except Exception as e:
                    print(f"Warning: Could not decode/save audio: {e}")
            
            # Append to chat history
            if text_out:
                self.chat_state.append(
                    text=torch.stack(text_out, 1) if len(text_out) > 1 else text_out[0].unsqueeze(1),
                    audio_out=torch.stack(audio_out, 1) if audio_out else None,
                    modality_flag=torch.tensor(modality_out)
                )
            
            self.chat_state.end_turn()
            self.conversation_turns += 1
            
            return AudioResponse(
                text=response_text,
                audio_tokens=audio_out,
                waveform=waveform,
                audio_path=saved_path
            )
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            raise
    
    def reset_conversation(self):
        """Clear chat history and reinitialize"""
        self.chat_state = ChatState(self.processor)
        self._setup_system_prompt()
        self.conversation_turns = 0
        print("Conversation reset.")