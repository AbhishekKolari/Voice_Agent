# # """Configuration settings for Voice Dictation Agent"""

# # import torch
# # from pathlib import Path

# # class Config:
# #     # Model settings
# #     HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
# #     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
# #     # Generation parameters
# #     MAX_NEW_TOKENS = 512
# #     AUDIO_TEMPERATURE = 1.0
# #     AUDIO_TOP_K = 4
# #     TEXT_TEMPERATURE = 0.7
# #     TEXT_TOP_P = 0.9
    
# #     # Audio settings
# #     SAMPLE_RATE = 24000  # Mimi codec output
    
# #     # Memory settings
# #     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# #     VECTOR_STORE_K = 3  # Number of similar conversations to retrieve
# #     RECENT_HISTORY_K = 5  # Number of recent turns to include
    
# #     # Paths
# #     BASE_DIR = Path(__file__).parent
# #     AUDIO_OUTPUT_DIR = BASE_DIR / "outputs" / "audio"
    
# #     # System prompts
# #     SYSTEM_PROMPT = (
# #         "You are a helpful voice assistant. Respond clearly and concisely. "
# #         "When appropriate, provide natural conversational responses."
# #     )
    
# #     DICTATION_SYSTEM_PROMPT = (
# #         "You are a voice dictation assistant. Transcribe the audio accurately "
# #         "and format it appropriately."
# #     )
    
# #     @classmethod
# #     def setup_directories(cls):
# #         """Create necessary directories"""
# #         cls.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # # Initialize directories on import
# # Config.setup_directories()

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
#     MAX_NEW_TOKENS = 512
#     AUDIO_TEMPERATURE = 1.0
#     AUDIO_TOP_K = 4
#     TEXT_TEMPERATURE = 0.7
#     TEXT_TOP_P = 0.9
    
#     # Audio settings
#     SAMPLE_RATE = 24000  # Mimi codec output
    
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
    # Note: Liquid AI's generate_interleaved and generate_sequential only support
    # audio_temperature and audio_top_k (not temperature/top_p for text)
    MAX_NEW_TOKENS = 512
    
    # Audio generation parameters
    AUDIO_TEMPERATURE = 1.0  # Controls randomness in audio generation (0.0 = deterministic, higher = more random)
    AUDIO_TOP_K = 4          # Only sample from top K audio tokens (lower = more focused, higher = more diverse)
    
    # Audio settings
    SAMPLE_RATE = 24000  # Mimi codec output sample rate
    
    # Memory settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_K = 3  # Number of similar conversations to retrieve
    RECENT_HISTORY_K = 5  # Number of recent turns to include
    
    # Paths
    BASE_DIR = Path(__file__).parent
    AUDIO_OUTPUT_DIR = BASE_DIR / "outputs" / "audio"
    
    # System prompts
    SYSTEM_PROMPT = (
        "You are a helpful voice assistant. Respond clearly and concisely. "
        "When appropriate, provide natural conversational responses."
    )
    
    DICTATION_SYSTEM_PROMPT = (
        "You are a voice dictation assistant. Transcribe the audio accurately "
        "and format it appropriately."
    )
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        cls.AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
Config.setup_directories()