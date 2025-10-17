"""
Main entry point for Voice Dictation Agent
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.voice_agent import VoiceDictationAgent
from src.utils.audio_utils import AudioRecorder, AudioPlayer
from config import Config


def print_menu():
    """Print the main menu"""
    print("\n" + "="*60)
    print("Voice Dictation Agent - Main Menu")
    print("="*60)
    print("1. Text Chat (type to talk)")
    print("2. Transcribe Audio File (speech-to-text)")
    print("3. Text-to-Speech (generate audio from text)")
    print("4. Voice Conversation (audio in, audio out)")
    print("5. Record Audio (from microphone)")
    print("6. Search Past Conversations")
    print("7. View Recent History")
    print("8. Reset Agent")
    print("9. Save/Load Memory")
    print("0. Exit")
    print("="*60)


def text_chat_mode(agent: VoiceDictationAgent):
    """Interactive text chat mode"""
    print("\n--- Text Chat Mode ---")
    print("Type your messages (or 'back' to return to menu)\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['back', 'exit', 'quit']:
            break
        
        if not user_input:
            continue
        
        try:
            response = agent.chat(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def transcribe_audio_mode(agent: VoiceDictationAgent):
    """Transcribe an audio file"""
    print("\n--- Transcribe Audio Mode ---")
    audio_path = input("Enter path to audio file: ").strip()
    
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        return
    
    try:
        transcription = agent.transcribe(audio_path)
        print(f"\n[Transcription]:\n{transcription}\n")
    except Exception as e:
        print(f"Error: {e}")


def text_to_speech_mode(agent: VoiceDictationAgent):
    """Convert text to speech"""
    print("\n--- Text-to-Speech Mode ---")
    text = input("Enter text to convert to speech: ").strip()
    
    if not text:
        print("No text provided")
        return
    
    try:
        audio_response = agent.text_to_speech(text)
        print(f"\n[Generated Audio]: {audio_response.audio_path}")
        
        play = input("Play audio now? (y/n): ").strip().lower()
        if play == 'y':
            AudioPlayer.play(audio_response.audio_path)
    except Exception as e:
        print(f"Error: {e}")


def voice_conversation_mode(agent: VoiceDictationAgent):
    """Voice conversation with audio input and output"""
    print("\n--- Voice Conversation Mode ---")
    audio_path = input("Enter path to audio file: ").strip()
    
    if not Path(audio_path).exists():
        print(f"Error: File not found: {audio_path}")
        return
    
    try:
        result = agent.process_audio_conversation(
            audio_path,
            generate_voice_response=True
        )
        
        print(f"\n[Transcription]: {result['transcription']}")
        print(f"[Response]: {result['response_text']}")
        print(f"[Audio Response]: {result['response_audio']}\n")
        
        play = input("Play response audio? (y/n): ").strip().lower()
        if play == 'y' and result['response_audio']:
            AudioPlayer.play(result['response_audio'])
    except Exception as e:
        print(f"Error: {e}")


def record_audio_mode(agent: VoiceDictationAgent):
    """Record audio from microphone"""
    print("\n--- Record Audio Mode ---")
    print("1. Record fixed duration")
    print("2. Record with manual stop (requires threading)")
    
    choice = input("Choose option: ").strip()
    
    recorder = AudioRecorder()
    
    if choice == '1':
        try:
            duration = float(input("Enter duration in seconds: ").strip())
            audio_path = recorder.record_fixed_duration(duration)
            
            if audio_path:
                action = input("\nWhat would you like to do?\n1. Transcribe\n2. Just save\nChoice: ").strip()
                if action == '1':
                    transcription = agent.transcribe(audio_path)
                    print(f"\n[Transcription]:\n{transcription}\n")
        except ValueError:
            print("Invalid duration")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Manual stop mode requires threading - not implemented in basic version")


def search_memory_mode(agent: VoiceDictationAgent):
    """Search past conversations"""
    print("\n--- Search Memory ---")
    query = input("Enter search query: ").strip()
    
    if not query:
        print("No query provided")
        return
    
    try:
        results = agent.search_past_conversations(query, k=3)
        
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} relevant conversations:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content}")
            print(f"   Timestamp: {doc.metadata.get('timestamp', 'N/A')}\n")
    except Exception as e:
        print(f"Error: {e}")


def view_history_mode(agent: VoiceDictationAgent):
    """View recent conversation history"""
    print("\n--- Recent History ---")
    try:
        history = agent.memory.get_recent_history(k=10)
        print(f"\n{history}\n")
    except Exception as e:
        print(f"Error: {e}")


def memory_management_mode(agent: VoiceDictationAgent):
    """Save or load memory"""
    print("\n--- Memory Management ---")
    print("1. Save memory to disk")
    print("2. Load memory from disk")
    
    choice = input("Choose option: ").strip()
    
    if choice == '1':
        path = input("Enter save path (default: ./memory): ").strip()
        path = path or "./memory"
        try:
            agent.save_memory(path)
        except Exception as e:
            print(f"Error: {e}")
    elif choice == '2':
        path = input("Enter load path: ").strip()
        if Path(path).exists():
            try:
                agent.load_memory(path)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Error: Path not found: {path}")


def run_examples(agent: VoiceDictationAgent):
    """Run example interactions"""
    print("\n" + "="*60)
    print("Running Example Interactions")
    print("="*60 + "\n")
    
    try:
        # Example 1: Text chat
        print("Example 1: Text conversation")
        response1 = agent.chat("What is the capital of France?")
        print(f"Q: What is the capital of France?")
        print(f"A: {response1}\n")
        
        # Example 2: Follow-up (tests memory)
        print("Example 2: Follow-up question (tests memory)")
        response2 = agent.chat("What is the population of that city?")
        print(f"Q: What is the population of that city?")
        print(f"A: {response2}\n")
        
        # Example 3: Search memory
        print("Example 3: Searching memory")
        search_results = agent.search_past_conversations("France capital", k=2)
        print(f"Search query: 'France capital'")
        print(f"Found {len(search_results)} relevant conversations\n")
        
        # Example 4: TTS
        print("Example 4: Text-to-Speech")
        audio_resp = agent.text_to_speech(
            "Hello! This is a test of the text to speech system.",
        )
        print(f"Generated audio: {audio_resp.audio_path}\n")
        
        print("="*60)
        print("Examples Complete!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error during examples: {e}")


def main():
    """Main application loop"""
    print("\n" + "="*60)
    print("Voice Dictation Agent")
    print("Powered by Liquid AI LFM2-Audio-1.5B")
    print("="*60 + "\n")
    
    # Check if user wants to run examples
    run_demo = input("Run example interactions? (y/n): ").strip().lower()
    
    # Initialize agent
    try:
        agent = VoiceDictationAgent()
    except Exception as e:
        print(f"\nError initializing agent: {e}")
        print("Please ensure:")
        print("1. All dependencies are installed: pip install -r requirements/base.txt")
        print("2. You have internet connection (to download the model)")
        print("3. You have sufficient GPU/CPU memory")
        return
    
    if run_demo == 'y':
        run_examples(agent)
    
    # Main menu loop
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip()
        
        try:
            if choice == '1':
                text_chat_mode(agent)
            elif choice == '2':
                transcribe_audio_mode(agent)
            elif choice == '3':
                text_to_speech_mode(agent)
            elif choice == '4':
                voice_conversation_mode(agent)
            elif choice == '5':
                record_audio_mode(agent)
            elif choice == '6':
                search_memory_mode(agent)
            elif choice == '7':
                view_history_mode(agent)
            elif choice == '8':
                confirm = input("Reset agent? This will clear all memory. (y/n): ").strip().lower()
                if confirm == 'y':
                    agent.reset()
            elif choice == '9':
                memory_management_mode(agent)
            elif choice == '0':
                print("\nExiting... Goodbye!\n")
                break
            else:
                print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()