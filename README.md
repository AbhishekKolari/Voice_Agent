<h1 align="center"> 🎙️ Voice Dictation Agent </h1>

<div align="center">

A multimodal AI agent powered by [Liquid AI's LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B) model and LangChain framework. This agent supports voice transcription (ASR), text-to-speech (TTS), and multi-turn audio conversations with semantic memory.

</div>

## 🌟 Features

    ### Core Capabilities

    - **🎤 Pure ASR (Automatic Speech Recognition)**: High-quality audio transcription using Liquid AI's foundation model
    - **🔊 Pure TTS (Text-to-Speech)**: Natural voice synthesis with customizable voice characteristics
    - **💬 Multi-turn Conversations**: Context-aware dialogue with conversation history
    - **🎵 Audio-to-Audio Conversations**: Direct audio input with audio response generation
    - **🧠 Semantic Memory**: FAISS vector store for searching past conversations
    - **🔗 LangChain Integration**: Extensible tool-based architecture

    ### Technical Features

    - **Three Generation Modes**:
    - `generate_sequential`: For single-turn ASR and TTS
    - `generate_interleaved`: For multi-turn multimodal conversations
    - **Memory Management**: Dual-tier memory (FAISS + conversation buffer)
    - **Automatic State Management**: Prevents conversation state corruption
    - **Error Recovery**: Graceful handling of CUDA errors and generation failures

## 💻 System Requirements

    ### Minimum Requirements

    | Component | Requirement |
    |-----------|-------------|
    | **Python** | 3.9 or higher |
    | **GPU** | NVIDIA GPU with 8GB+ VRAM (CUDA-capable) |
    | **RAM** | 16GB+ |
    | **Storage** | 5GB+ (for model weights) |
    | **CUDA** | 11.8 or 12.1+ |
    | **PyTorch** | 2.8.0 or higher |

    ### Recommended Setup

    - **GPU**: NVIDIA RTX 3090/4090 or A100
    - **RAM**: 32GB+
    - **VRAM**: 16GB+

    ### ⚠️ Important Notes

    - **CPU-only mode is NOT supported** by the Liquid AI model
    - The model requires significant GPU memory (~3GB for model weights + additional for inference)
    - Generation can be slow on consumer GPUs (5-30 seconds per response)
    - **Running locally may cause OOM errors** on GPUs with <8GB VRAM

## 🚀 Installation

### Option 1: Local Installation (Recommended for GPU Users)

#### Setup Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/AbhishekKolari/Voice_Agent.git
    cd voice-dictation-agent
    ```

2. **Create virtual environment**:

    ```bash
    # create the environment
    py -3.13 -m venv [env_name]

    # activate 
    source [env_name]/bin/activate   # Windows: [env_name]\Scripts\activate

    # install dependencies
    pip install -r requirements/base.txt
    ```

3. **Install PyTorch with CUDA**:

    Verify CUDA Installation
    ```bash
    nvidia-smi
    ```

    Check PyTorch official site
    Go to https://pytorch.org/get-started/locally/ and select:

    - PyTorch Build: Stable or Nightly
    - Your OS: Windows
    - Package: Pip
    - Language: Python
    - Compute Platform: CUDA 12.1 (or your version)

    It will give you the exact command.

4. **Verify installation**:

    ```bash
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

### Option 2: Google Colab (Recommended for Testing)

    Perfect for users without local GPU access:

    ```python
    # In a Colab notebook
    !git clone https://github.com/AbhishekKolari/Voice_Agent.git
    %cd Voice_Agent

    # Install dependencies
    !pip install -r requirements/base.txt

    # Install Port_audio
    !apt install libportaudio2

    # Test loading model
    !python liquid_llm.py

    # Run the agent
    !python main.py
    ```

    **Colab Advantages**:
    - ✅ Free GPU access (T4/V100)
    - ✅ No local setup required
    - ✅ Pre-installed CUDA environment

    **Colab Limitations**:
    - ❌ No microphone recording
    - ❌ Session timeouts (need to re-run)
    - ❌ Limited to ~12 hours per session

## ▶️ Usage

    Run the agent
    ```bash
    python main.py  # Colab: !python main.py
    ```
