# main.py

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

import asyncio
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import time
import threading
from agents import Runner
from datetime import datetime

from stt_tts_loader import stt_model, processor, tts_model, device
from multiagent import conversation_agent, UserContext, format_for_voice


class VoiceInteractionManager:
    def __init__(self, chunk_duration: float = 5.0):
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_playing_response = False
        self.mic_stream = None
        
        # Initialize microphone stream
        self.setup_microphone()
        
    def setup_microphone(self):
        """Setup microphone stream"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            # Only collect audio when actively recording
            if self.is_recording and not self.is_playing_response:
                self.audio_queue.put(indata.copy())
        
        self.mic_stream = sd.InputStream(
            callback=audio_callback, 
            samplerate=16000, 
            channels=1,
            blocksize=1024
        )
    
    def start_recording(self):
        """Start recording audio"""
        print("🎤 Listening... (speak for up to {} seconds)".format(self.chunk_duration))
        
        # Clear any old audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.is_recording = True
        if not self.mic_stream.active:
            self.mic_stream.start()
    
    def stop_recording(self):
        """Stop recording and return collected audio"""
        self.is_recording = False
        
        # Collect all audio from queue
        buffer = []
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                buffer.append(chunk)
            except queue.Empty:
                break
        
        if buffer:
            audio_data = np.concatenate(buffer, axis=0).flatten()
            return audio_data
        return None
    
    def transcribe_audio(self, audio_data):
        """Convert audio to text using STT model"""
        if audio_data is None or len(audio_data) == 0:
            return ""
        
        try:
            print("🔄 Transcribing...")
            
            # Prepare audio for Whisper
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="english", task="transcribe"
            )

            input_features = processor(
                audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = stt_model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )

            transcription = processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""
    
    async def get_agent_response(self, user_input, user_context):
        """Get response from multiagent system"""
        try:
            print(f"🤔 Processing: '{user_input}'")
            
            # Format conversation history for the prompt (only completed conversations)
            history_context = ""
            if user_context.conversation_history:  # If there's any previous history
                history_context = "\n\nPrevious conversation:\n"
                for entry in user_context.conversation_history:
                    if entry["assistant"]:  # Only include completed exchanges
                        history_context += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
            
            # Create enhanced input with context
            enhanced_input = f"{user_input}{history_context}"
            
            result = await Runner.run(conversation_agent, enhanced_input, context=user_context)
            
            # Extract response - handle both structured and regular outputs
            if hasattr(result, "final_output"):
                raw_response = result.final_output
            else:
                raw_response = result
            
            # Format the response for voice output
            formatted_response = format_for_voice(raw_response)
            
            # Add to conversation history using the dataclass method
            user_context.add_to_history(user_input, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            print(f"❌ Agent processing error: {e}")
            return "Sorry, I encountered an error processing your request."
            
    
    def speak_response(self, text):
        """Convert text to speech and play it"""
        try:
            # Truncate very long responses for better user experience
            if len(text) > 500:
                text = text[:450] + "... I can provide more details if you'd like."
            
            print(f"🔊 Speaking: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            # Set flag to prevent recording during TTS playback
            self.is_playing_response = True
            
            # Generate speech
            tts_model.tts_to_file(text=text, file_path="response.wav")
            
            # Play audio
            audio_out, sr = sf.read("response.wav")
            sd.play(audio_out, samplerate=sr)
            sd.wait()  # Wait for playback to complete
            
            # Small delay to ensure audio is fully finished
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            # Fallback: at least print the text
            print(f"📢 Text response: {text}")
        finally:
            self.is_playing_response = False
    
    def cleanup(self):
        """Clean up resources"""
        self.is_recording = False
        if self.mic_stream and self.mic_stream.active:
            self.mic_stream.stop()
            self.mic_stream.close()


# --- Automatic voice loop with improved conversation handling ---

async def automatic_voice_loop():
    """Automatic mode with progressive timeout - eventually gives up"""
    
    voice_manager = VoiceInteractionManager(chunk_duration=5.0)
    user_context = UserContext(user_id="user001")
    
    print("🎙️ Automatic Voice Mode with Smart Timeout")
    voice_manager.speak_response("Hello! I'm your AI assistant. I can help you with trending news, fact-checking, and general conversation. What would you like to talk about?")
    
    # Timeout configuration
    consecutive_empty = 0
    max_empty_cycles = 3        # Before first pause (9 seconds)
    pause_count = 0
    max_pauses = 2              # Maximum number of pauses before giving up
    pause_durations = [5, 10, 20]  # Progressive pause lengths
    
    try:
        while True:
            print(f"\n🎤 Listening automatically... (Pause {pause_count}/{max_pauses})")
            
            # Record audio
            voice_manager.start_recording()
            await asyncio.sleep(voice_manager.chunk_duration)
            audio_data = voice_manager.stop_recording()
            
            if audio_data is not None:
                transcription = voice_manager.transcribe_audio(audio_data)
                
                if transcription:
                    # Got speech - reset all counters
                    consecutive_empty = 0
                    pause_count = 0
                    
                    print(f"📝 You said: '{transcription}'")
                    
                    # Check for exit
                    if any(phrase in transcription.lower() for phrase in 
                          ['stop listening', 'goodbye', 'quit', 'exit', 'shut down']):
                        voice_manager.speak_response("It was great chatting with you! Have a wonderful day!")
                        break

                    # Process and respond
                    response_text = await voice_manager.get_agent_response(
                        transcription, user_context
                    )
                    
                    if response_text:
                        print(f"🤖 Agent: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
                        voice_manager.speak_response(response_text)
                        await asyncio.sleep(1)  # Brief pause before listening again
                
                else:
                    # No speech detected
                    consecutive_empty += 1
                    print(f"🔇 Silent cycle {consecutive_empty}/{max_empty_cycles}")
                    
                    if consecutive_empty >= max_empty_cycles:
                        # Time for a pause
                        consecutive_empty = 0
                        
                        if pause_count >= max_pauses:
                            # Too many pauses - go to sleep mode
                            print("😴 Going to sleep mode due to extended silence...")
                            voice_manager.speak_response(
                                "I haven't heard from you in a while, so I'm going to sleep mode. "
                                "Say 'wake up' or restart the program when you want to chat again."
                            )
                            
                            # Enter deep sleep mode
                            await enter_sleep_mode(voice_manager, user_context)
                            break
                            
                        else:
                            # Pause with increasing duration
                            pause_duration = pause_durations[min(pause_count, len(pause_durations)-1)]
                            pause_count += 1
                            
                            print(f"😴 Pausing for {pause_duration} seconds... (Pause {pause_count}/{max_pauses})")
                            
                            # More natural pause messages
                            pause_messages = [
                                "I'm still here when you're ready to continue our conversation.",
                                "Take your time. I'll wait a bit longer for you.",
                                "Still listening. Let me know if you need anything, or say 'stop listening' to exit."
                            ]
                            pause_msg = pause_messages[min(pause_count-1, len(pause_messages)-1)]
                            voice_manager.speak_response(pause_msg)
                            
                            await asyncio.sleep(pause_duration)
            
            # Small delay between cycles
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n👋 Automatic mode interrupted")
    except Exception as e:
        print(f"❌ Error in automatic mode: {e}")
    finally:
        voice_manager.cleanup()


# --- Sleep mode function ---

async def enter_sleep_mode(voice_manager, user_context):
    """Deep sleep mode - only wake on specific phrases"""
    
    print("💤 Entering sleep mode...")
    print("Say 'wake up', 'hello', or 'hey assistant' to wake me up")
    
    wake_words = ['wake up', 'hello', 'hey assistant', 'are you there', 'wake']
    sleep_check_interval = 5  # Check every 5 seconds
    
    while True:
        try:
            # Listen for wake words
            voice_manager.start_recording()
            await asyncio.sleep(sleep_check_interval)
            audio_data = voice_manager.stop_recording()
            
            if audio_data is not None:
                transcription = voice_manager.transcribe_audio(audio_data)
                
                if transcription:
                    transcription_lower = transcription.lower()
                    print(f"💤 Sleep mode heard: '{transcription}'")
                    
                    # Check for wake words
                    if any(wake_word in transcription_lower for wake_word in wake_words):
                        print("🌅 Waking up!")
                        voice_manager.speak_response(
                            "Hello again! I'm back and ready to continue our conversation. What would you like to talk about?"
                        )
                        
                        # Return to normal operation
                        await automatic_voice_loop()
                        break
                    
                    # Check for exit commands even in sleep mode
                    elif any(phrase in transcription_lower for phrase in 
                            ['quit', 'exit', 'goodbye', 'shut down']):
                        voice_manager.speak_response("Shutting down. Goodbye!")
                        break
                        
                    else:
                        # Ignore other speech in sleep mode
                        print("💤 Still sleeping... (say 'wake up' to wake me)")
            
            await asyncio.sleep(1)  # Prevent busy waiting
            
        except KeyboardInterrupt:
            print("\n👋 Sleep mode interrupted")
            break
        except Exception as e:
            print(f"❌ Error in sleep mode: {e}")
            break


# --- Main Function with Mode Selection ---
async def main():
    print("🎙️ AI Voice Assistant")
    print("=" * 30)
    
    try:
        await automatic_voice_loop()
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())