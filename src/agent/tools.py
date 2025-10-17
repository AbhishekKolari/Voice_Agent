"""
LangChain tools for the voice agent
"""
from typing import List
from langchain.agents import Tool

from src.models.liquid_llm import LiquidAudioLLM
from src.memory.vector_memory import VoiceAgentMemory


def create_agent_tools(llm: LiquidAudioLLM, memory: VoiceAgentMemory) -> List[Tool]:
    """
    Create tools for the voice agent
    
    Args:
        llm: Liquid AI LLM instance
        memory: Memory instance
        
    Returns:
        List of LangChain tools
    """
    
    def search_memory_tool(query: str) -> str:
        """Search through past conversations"""
        results = memory.search_similar(query, k=3)
        if not results:
            return "No relevant past conversations found."
        
        formatted = "Relevant past conversations:\\n"
        for i, doc in enumerate(results, 1):
            formatted += f"\\n{i}. {doc.page_content}\\n"
        return formatted
    
    def transcribe_audio_tool(audio_path: str) -> str:
        """Transcribe an audio file to text"""
        try:
            transcription = llm.transcribe_audio(audio_path)
            return f"Transcription: {transcription}"
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"
    
    def get_recent_context_tool(dummy_input: str = "") -> str:
        """Get recent conversation context"""
        return memory.get_recent_history(k=5)
    
    tools = [
        Tool(
            name="SearchMemory",
            func=search_memory_tool,
            description=(
                "Search through past conversations to find relevant information. "
                "Input should be a search query about past interactions. "
                "Use this when the user refers to previous conversations."
            )
        ),
        Tool(
            name="TranscribeAudio",
            func=transcribe_audio_tool,
            description=(
                "Transcribe an audio file to text. "
                "Input should be the file path to the audio file. "
                "Use this for speech-to-text conversion."
            )
        ),
        Tool(
            name="GetRecentContext",
            func=get_recent_context_tool,
            description=(
                "Get the recent conversation history. "
                "Use this to understand the context of the current conversation. "
                "No input required."
            )
        ),
    ]
    
    return tools