""" 
Main Voice Dictation Agent class
"""
from typing import Optional
from pathlib import Path

from src.models.liquid_llm import LiquidAudioLLM, AudioResponse
from src.memory.vector_memory import VoiceAgentMemory
from src.agent.tools import create_agent_tools


class VoiceDictationAgent:
    """
    Main agent class combining Liquid AI model with LangChain tools and memory
    
    Modes:
    1. Dictation Mode: Pure speech-to-text transcription
    2. Conversation Mode: Multi-turn dialogue with context
    3. Voice Response Mode: Generate spoken responses
    """
    
    def __init__(self):
        print("\\n" + "="*60)
        print("Initializing Voice Dictation Agent...")
        print("="*60 + "\\n")
        
        self.llm = LiquidAudioLLM()
        self.memory = VoiceAgentMemory()
        self.tools = create_agent_tools(self.llm, self.memory)
        
        print("\\n" + "="*60)
        print("Voice Dictation Agent Ready!")
        print("="*60 + "\\n")
    
    def transcribe(self, audio_path: str, save_to_memory: bool = True) -> str:
        """
        Transcribe audio to text (Dictation Mode)
        Uses generate_sequential for single-turn transcription
        
        Args:
            audio_path: Path to audio file
            save_to_memory: Whether to save transcription to memory
            
        Returns:
            Transcribed text
        """
        print(f"\\n[Transcribing]: {audio_path}")
        transcription = self.llm.transcribe_audio(audio_path)
        
        if save_to_memory:
            self.memory.add_interaction(
                user_input=f"[Audio transcription from {Path(audio_path).name}]",
                assistant_response=transcription
            )
        
        return transcription
    
    def chat(self, user_input: str, use_memory: bool = True) -> str:
        """
        Text-based conversation (Conversation Mode)
        Uses generate_interleaved for multi-turn dialogue
        
        Args:
            user_input: User's text input
            use_memory: Whether to use conversation history
            
        Returns:
            Assistant's text response
        """
        # Get recent context if using memory
        context = ""
        if use_memory:
            recent_history = self.memory.get_recent_history(k=3)
            if recent_history != "No recent history":
                context = f"Recent conversation:\\n{recent_history}\\n\\n"
        
        # Enhance prompt with context
        enhanced_prompt = f"{context}Current input: {user_input}"
        
        # Generate response using the LLM's _call method (interleaved)
        response = self.llm._call(enhanced_prompt)
        
        # Save to memory
        if use_memory:
            self.memory.add_interaction(user_input, response)
        
        return response
    
    def chat_with_voice(
        self, 
        user_input: str, 
        output_path: Optional[str] = None,
        use_memory: bool = True
    ) -> AudioResponse:
        """
        Conversation with voice response (Voice Response Mode)
        Uses generate_interleaved for multi-turn with audio output
        
        Args:
            user_input: User's text input
            output_path: Path to save audio response
            use_memory: Whether to use conversation history
            
        Returns:
            AudioResponse with text and audio
        """
        # Get recent context if using memory
        if use_memory:
            recent_history = self.memory.get_recent_history(k=3)
            if recent_history != "No recent history":
                # Inject context into chat state before generation
                pass  # Context is maintained in chat_state
        
        # Generate audio response
        audio_response = self.llm.chat_with_audio_response(user_input, output_path)
        
        # Save to memory
        if use_memory:
            self.memory.add_interaction(user_input, audio_response.text)
        
        return audio_response
    
    def process_audio_conversation(
        self, 
        audio_path: str,
        generate_voice_response: bool = False,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Process audio input and generate response
        
        Args:
            audio_path: Path to input audio
            generate_voice_response: Whether to generate audio response
            output_path: Path for output audio
            
        Returns:
            Dictionary with transcription and response
        """  
        # Transcribe audio
        transcription = self.transcribe(audio_path, save_to_memory=False)
        print(f"[Transcribed]: {transcription}\\n")
        
        # Generate response
        if generate_voice_response:
            audio_response = self.chat_with_voice(
                transcription, 
                output_path=output_path
            )
            return {
                "transcription": transcription,
                "response_text": audio_response.text,
                "response_audio": audio_response.audio_path
            }
        else:
            response = self.chat(transcription)
            return {
                "transcription": transcription,
                "response_text": response,
                "response_audio": None
            }
    
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> AudioResponse:
        """
        Convert text to speech (TTS Mode)
        Uses generate_sequential for single-turn TTS
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio
            
        Returns:
            AudioResponse with generated audio
        """
        return self.llm.generate_audio_response(text, output_path)
    
    def search_past_conversations(self, query: str, k: int = 3) -> list:
        """Search through past conversations"""
        return self.memory.search_similar(query, k=k)
    
    def reset(self):
        """Reset conversation state and memory"""
        self.llm.reset_conversation()
        self.memory.clear()
        print("\\nAgent reset complete.\\n")
    
    def save_memory(self, path: str):
        """Save memory to disk"""
        self.memory.save_to_disk(path)
    
    def load_memory(self, path: str):
        """Load memory from disk"""
        self.memory.load_from_disk(path)