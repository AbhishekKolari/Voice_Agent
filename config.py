# """
# Configuration settings for Voice Dictation Agent
# """
# import torch
# from pathlib import Path

# class Config:
#     # Model settings
#     HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
    
#     # Device selection - Force CPU if CUDA not properly configured
#     try:
#         if torch.cuda.is_available():
#             # Test if CUDA actually works
#             torch.zeros(1).cuda()
#             DEVICE = "cuda"
#         else:
#             DEVICE = "cpu"
#     except:
#         DEVICE = "cpu"
    
#     # Generation parameters
#     # Note: Liquid AI's generate_interleaved and generate_sequential only support
#     # audio_temperature and audio_top_k (not temperature/top_p for text)
#     MAX_NEW_TOKENS = 512
    
#     # Audio generation parameters
#     AUDIO_TEMPERATURE = 1.0  # Controls randomness in audio generation (0.0 = deterministic, higher = more random)
#     AUDIO_TOP_K = 4          # Only sample from top K audio tokens (lower = more focused, higher = more diverse)
    
#     # Audio settings
#     SAMPLE_RATE = 24000  # Mimi codec output sample rate
    
#     # Memory settings
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#     VECTOR_STORE_K = 3  # Number of similar conversations to retrieve
#     RECENT_HISTORY_K = 5  # Number of recent turns to include
    
#     # Paths
#     BASE_DIR = Path(__file__).parent
#     AUDIO_OUTPUT_DIR = BASE_DIR / "outputs" / "audio"
    
#     # System prompts
#     SYSTEM_PROMPT = (
#         "You are a helpful voice assistant. Respond clearly and concisely. "
#         "When appropriate, provide natural conversational responses."
#     )
    
#     DICTATION_SYSTEM_PROMPT = (
#         "You are a voice dictation assistant. Transcribe the audio accurately "
#         "and format it appropriately."
#     )
    
#     @classmethod
#     def setup_directories(cls):
#         """Create necessary directories"""
#         cls.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Initialize directories on import
# Config.setup_directories()

"""
Configuration settings for Voice Dictation Agent
"""
import torch
from pathlib import Path

class Config:
    # Model settings
    HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
    
    # Device selection - Force CPU if CUDA not properly configured
    try:
        if torch.cuda.is_available():
            # Test if CUDA actually works
            torch.zeros(1).cuda()
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
    except:
        DEVICE = "cpu"
    
    # Generation parameters
    MAX_NEW_TOKENS = 512
    
    # Audio generation parameters (for multi-turn chat)
    AUDIO_TEMPERATURE = 1.0  # Controls randomness in audio generation
    AUDIO_TOP_K = 4          # Only sample from top K audio tokens
    
    # TTS-specific parameters (following official Liquid AI examples)
    TTS_AUDIO_TEMPERATURE = 0.8  # Lower for more consistent TTS
    TTS_AUDIO_TOP_K = 64         # Higher for more diverse speech
    
    # Audio settings
    SAMPLE_RATE = 24000  # Mimi codec output sample rate
    
    # Memory settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_K = 3  # Number of similar conversations to retrieve
    RECENT_HISTORY_K = 5  # Number of recent turns to include
    
    # Paths
    BASE_DIR = Path(__file__).parent
    AUDIO_OUTPUT_DIR = BASE_DIR / "outputs" / "audio"
    
    # System prompts (following official Liquid AI patterns)
    SYSTEM_PROMPT = "Respond with interleaved text and audio."
    ASR_SYSTEM_PROMPT = "Perform ASR."
    TTS_SYSTEM_PROMPT = "Perform TTS."
    
    # Default voice description for TTS
    DEFAULT_VOICE = (
        "A clear speaker delivers the text with a natural tone. "
        "The recording is of good quality with minimal noise."
    )
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
Config.setup_directories()