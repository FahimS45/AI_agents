# 🎙️ Multiagent Voice-Integrated AI System

An intelligent voice-powered AI assistant that combines speech-to-text, text-to-speech, and multiagent AI capabilities for real-time news retrieval and fact-checking through natural voice interactions.

## ✨ Features

- **🎤 Real-time Voice Processing**: Automatic speech recognition with intelligent pause detection
- **🗣️ Natural Text-to-Speech**: High-quality voice synthesis for responses
- **🤖 Multiagent Architecture**: Specialized AI agents for different tasks
- **📰 Trending News Retrieval**: Real-time news fetching across various categories
- **✅ Fact-Checking**: Automated claim verification using web-sourced evidence
- **😴 Smart Sleep Mode**: Energy-efficient operation with wake-word activation
- **🔄 Automatic Conversation Flow**: Seamless voice interaction with progressive timeout

## 🏗️ Architecture

The system uses a sophisticated multiagent architecture with three specialized agents:

1. **Conversation Controller Agent**: Main orchestrator that manages user intent and routes requests
2. **Trending News Agent**: Specialized in fetching and formatting real-time news headlines
3. **Fact Checker Agent**: Verifies claims using retrieved web documents and provides evidence-based verdicts

## 🛠️ Technology Stack

### AI Models Used
- **Speech-to-Text**: OpenAI Whisper Base (`openai/whisper-base`)
- **Text-to-Speech**: Coqui TTS Tacotron2-DDC (`tts_models/en/ljspeech/tacotron2-DDC_ph`)
- **LLM Backend**: Configurable via environment variables (supports OpenAI-compatible APIs)

### Key Libraries
- **Agent Framework**: Custom agents framework with handoff capabilities
- **Audio Processing**: SoundDevice, SoundFile, NumPy
- **Web Search**: DuckDuckGo Search (DDGS)
- **ML/AI**: PyTorch, Transformers, TTS
- **Web Retrieval**: Custom RAG retriever system

### Hardware Requirements
- **Tested on**: NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
- **Minimum**: CUDA-compatible GPU with 6GB+ VRAM
- **CPU Fallback**: Supported but significantly slower
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for model downloads

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have Python 3.8+ and CUDA installed
python --version
nvcc --version
```

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multiagent-voice-ai.git
cd multiagent-voice-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
BASE_URL=your_openai_compatible_endpoint
API_KEY=your_api_key
MODEL_NAME=your_model_name
```

4. **Install system audio dependencies**
```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# On macOS
brew install portaudio

# On Windows
# PyAudio should install automatically with pip
```

### Running the System
```bash
python main.py
```

## 💬 Usage

### Voice Commands
- **General conversation**: Just speak naturally
- **Get trending news**: "What's the latest news?" or "Show me tech news"
- **Fact-checking**: "Is it true that..." or "Can you verify this claim..."
- **Exit commands**: "Stop listening", "Goodbye", "Quit", or "Exit"

### Sleep Mode
The system automatically enters sleep mode after periods of inactivity:
- **Wake commands**: "Wake up", "Hello", "Hey assistant", "Are you there"
- **Sleep behavior**: Progressive timeout system (3 cycles → pause → deeper sleep)

## 🔧 Configuration

### Audio Settings
```python
# In VoiceInteractionManager
chunk_duration = 3.0  # Recording chunk duration in seconds
samplerate = 16000     # Audio sample rate
channels = 1           # Mono audio
```

### Agent Customization
Modify agent instructions in `multiagent.py`:
```python
# Example: Customize trending agent behavior
trending_agent = Agent(
    instructions="Your custom instructions here...",
    # ... other parameters
)
```

### Model Selection
Update model configurations in `stt_tts_loader.py`:
```python
# Change Whisper model
whisper_model_name = "openai/whisper-large-v2"  # For better accuracy

# Change TTS model
tts_model = TTS("tts_models/en/vctk/vits")  # For different voices
```

## 📁 Project Structure

```
├── main.py                 # Main application entry point
├── multiagent.py           # Multiagent system configuration
├── stt_tts_loader.py       # Speech models loading
├── rag_retriever.py        # Web content retrieval system
├── llm_loader.py           # LLM loading utilities
├── agents/                 # Agent framework modules
├── .env                    # Environment configuration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔍 Core Components

### Voice Interaction Manager
- Handles real-time audio capture and playback
- Manages recording states and audio queuing
- Implements smart timeout and sleep mode logic

### Multiagent System
- **Function Tools**: Web search and fact-checking capabilities
- **Agent Handoffs**: Seamless task delegation between specialized agents
- **Structured Outputs**: Type-safe responses using Pydantic models

### Audio Pipeline
1. **Capture**: Continuous microphone monitoring
2. **Process**: Whisper-based speech recognition
3. **Analyze**: Intent classification and agent routing
4. **Respond**: LLM-generated responses
5. **Synthesize**: TTS audio generation
6. **Playback**: High-quality audio output

## 🎯 Use Cases

- **News Briefings**: Get personalized news updates across categories
- **Fact Verification**: Verify claims and statements with web evidence
- **Research Assistant**: Voice-powered information gathering
- **Accessibility Tool**: Hands-free information access
- **Smart Home Integration**: Voice-controlled information hub

## ⚠️ Known Limitations

- Requires stable internet connection for web search and fact-checking
- TTS playback prevents simultaneous speech recognition
- GPU memory usage scales with model size
- Wake word detection is phrase-based, not single-word optimized

## 🛠️ Troubleshooting

### Common Issues

**Audio not working:**
```bash
# Test audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**CUDA memory errors:**
- Reduce model sizes or switch to CPU mode
- Monitor GPU memory: `nvidia-smi`

**Import errors:**
```bash
# Reinstall audio dependencies
pip uninstall pyaudio sounddevice
pip install pyaudio sounddevice
```

**Agent errors:**
- Verify `.env` configuration
- Check API endpoint connectivity
- Ensure sufficient API credits/quota

## 🚧 Future Enhancements

- [ ] Multi-language support
- [ ] Custom wake word training
- [ ] Voice cloning capabilities
- [ ] Real-time streaming responses
- [ ] Mobile app integration
- [ ] Docker containerization
- [ ] Web interface dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech recognition model
- **Coqui TTS** for high-quality text-to-speech synthesis
- **Hugging Face** for model hosting and transformers library
- **Agent Framework** developers for the multiagent architecture

---

**Tested Environment**: NVIDIA GeForce RTX 3070 Ti (8GB VRAM) | Python 3.9+ | CUDA 11.8+

