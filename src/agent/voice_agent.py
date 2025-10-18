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
        print("\n" + "="*60)
        print("Initializing Voice Dictation Agent...")
        print("="*60 + "\n")
        
        self.llm = LiquidAudioLLM()
        self.memory = VoiceAgentMemory()
        self.tools = create_agent_tools(self.llm, self.memory)
        
        print("\n" + "="*60)
        print("Voice Dictation Agent Ready!")
        print("="*60 + "\n")
    
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
        print(f"\n[Transcribing]: {audio_path}")
        transcription = self.llm.transcribe_audio(audio_path)
        
        if save_to_memory and transcription:
            # Save the actual transcription, not the assistant response
            self.memory.add_interaction(
                user_input=transcription,  # Use transcription as user input
                assistant_response=f"[Transcribed from {Path(audio_path).name}]"
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
        # Note: We don't inject memory into prompt for interleaved mode
        # as the model maintains its own chat state
        # The memory is only for search/retrieval across sessions
        
        # Generate response using the LLM's _call method (interleaved)
        response = self.llm._call(user_input)
        
        # Save to memory for cross-session retrieval
        if use_memory and response:
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
        # Generate audio response
        audio_response = self.llm.chat_with_audio_response(user_input, output_path)
        
        # Save to memory
        if use_memory and audio_response.text:
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
        if generate_voice_response:
            # Use the new method that processes audio directly with generate_interleaved
            try:
                # First transcribe to get the text
                transcription = self.transcribe(audio_path, save_to_memory=False)
                print(f"[Transcribed]: {transcription}\n")
                
                # Then generate audio response using the direct audio input method
                audio_response = self.llm.process_audio_with_response(audio_path, output_path)
                
                # Save to memory
                self.memory.add_interaction(transcription, audio_response.text)
                
                return {
                    "transcription": transcription,
                    "response_text": audio_response.text,
                    "response_audio": audio_response.audio_path
                }
            except Exception as e:
                print(f"Error during audio conversation: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "transcription": "[Error]",
                    "response_text": f"Error: {str(e)}",
                    "response_audio": None
                }
        else:
            # Just transcribe and respond with text
            transcription = self.transcribe(audio_path, save_to_memory=False)
            print(f"[Transcribed]: {transcription}\n")
            
            if not transcription or transcription.strip() == "":
                return {
                    "transcription": "[No speech detected]",
                    "response_text": "I couldn't hear anything. Could you try again?",
                    "response_audio": None
                }
            
            try:
                response = self.chat(transcription)
                return {
                    "transcription": transcription,
                    "response_text": response,
                    "response_audio": None
                }
            except Exception as e:
                print(f"Error generating response: {e}")
                return {
                    "transcription": transcription,
                    "response_text": f"Error: {str(e)}",
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
        print(f"\n[Converting to speech]: {text[:50]}...")
        return self.llm.generate_audio_response(text, output_path)
    
    def search_past_conversations(self, query: str, k: int = 3) -> list:
        """
        Search through past conversations
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        results = self.memory.search_similar(query, k=k)
        
        # Filter out irrelevant results by checking if query words are in content
        query_words = set(query.lower().split())
        filtered_results = []
        
        for doc in results:
            content_words = set(doc.page_content.lower().split())
            # Check if at least one query word appears in content
            if query_words & content_words:
                filtered_results.append(doc)
        
        return filtered_results if filtered_results else results
    
    def reset(self):
        """Reset conversation state and memory"""
        self.llm.reset_conversation()
        self.memory.clear()
        print("\nAgent reset complete.\n")
    
    def save_memory(self, path: str):
        """Save memory to disk"""
        self.memory.save_to_disk(path)
    
    def load_memory(self, path: str):
        """Load memory from disk"""
        self.memory.load_from_disk(path)