# Nova.AI Web Interface

[![Nova.AI Demo](https://img.shields.io/badge/Demo-Live-blue)](https://your-demo-link.com)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)  
[![OpenAI](https://img.shields.io/badge/OpenAI-API-blue.svg)](https://platform.openai.com/docs/introduction)

Nova.AI is a real-time meeting transcription and AI analysis tool powered by OpenAI. It captures audio from microphones or system speakers, transcribes it in real-time, generates summaries, and provides intelligent response suggestions. The frontend is a modern web interface built with HTML, CSS, and JavaScript, while the backend is a high-performance FastAPI server.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

## Features

- **Real-time Audio Capture**: Record from microphone or system speakers (via screen/tab share).
- **Live Transcription**: Chunk-based transcription using OpenAI's Whisper model.
- **AI-Powered Analysis**:
  - Automatic summaries with key points and action items.
  - Intelligent response suggestions based on conversation context.
- **Visualizations**: Audio waveform, progress indicators, and status updates.
- **Caching & Optimization**: In-memory caching for repeated transcriptions.
- **Cross-Platform**: Works on desktop browsers; supports multiple microphones.
- **Secure & Efficient**: Uses FastAPI for backend, with CORS and GZip middleware.
- **Extensible**: Easy integration with other AI services.

## Demo

Check out a live demo [here](https://cheerful-pithivier-499ce7.netlify.app/) (replace with your hosted URL, e.g., on Render or Vercel).

![Nova.AI Screenshot](<img width="1918" height="1085" alt="image" src="https://github.com/user-attachments/assets/85f0d03e-5a98-4603-a578-d4a43544054c" />)  
*(Add a screenshot of the interface here for visual appeal.)*

## Architecture

- **Frontend**: Single-page HTML app with JavaScript for audio handling, WebRTC for recording, and Fetch API for backend communication.
- **Backend**: FastAPI server with OpenAI integration for transcription (`/transcribe`), summarization (`/summarize`), and suggestions (`/suggest_response`).
- **Data Flow**:
  1. Frontend records audio in 5-second chunks.
  2. Sends chunks to backend for transcription.
  3. Backend processes with OpenAI and returns results.
  4. Frontend updates UI with transcript, summary, and suggestions in real-time.

## Prerequisites

- Python 3.10+
- OpenAI API Key (sign up at [platform.openai.com](https://platform.openai.com))
- Browser with WebRTC support (Chrome, Firefox, Edge)
- Optional: FFmpeg for audio processing (if using pydub for conversions)

## Installation

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/nova-ai.git
   cd nova-ai
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install fastapi uvicorn openai pydub httpx
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PORT=8000  # Optional, default is 8000
   ENVIRONMENT=development  # Or 'production'
   ```

### Frontend Setup

The frontend is a static HTML file (`index.html`). No additional setup is required beyond serving it (e.g., via a web server or opening directly in a browser).

## Configuration

- **Backend URL**: In the frontend JavaScript (`NovaWebInterface` class), update `this.backendUrl` if not using the default (`https://openai-integrated-backend.onrender.com` or `http://localhost:8000`).
- **OpenAI Models**: The backend uses `whisper-1` for transcription and `gpt-4o-mini` for analysis. Update in the code if needed.
- **Audio Settings**: Adjust chunk size (default: 5 seconds) in `startRecording()` for better latency vs. accuracy.

## Usage

### Running the Backend

```
uvicorn app:app --reload --port 8000
```

Access the API docs at `http://localhost:8000/docs`.

### Running the Frontend

1. Open `index.html` in a browser.
2. Select microphone or enable speaker capture.
3. Click "Start Recording" to begin.
4. View live transcription, summary, and suggestions in the UI.

For production, deploy the backend to Render/Heroku and host the frontend on Vercel/Netlify.

### Example Workflow

1. Start recording a meeting.
2. Audio chunks are transcribed in real-time.
3. As transcript builds, summaries and suggestions update automatically.

## API Endpoints

- **GET /**: Root info.
- **GET /health**: Health check.
- **POST /transcribe**: Transcribe audio file (multipart form with `audio` field).
- **POST /summarize**: Summarize text (JSON body: `{ "text": "..." }`).
- **POST /suggest_response**: Get response suggestion (JSON body: `{ "text": "..." }`).

Full API docs available via Swagger at `/docs`.

## Troubleshooting

- **No Microphone Detected**: Ensure browser permissions are granted.
- **Backend Connection Issues**: Check console for errors; verify URL in frontend.
- **OpenAI Errors**: Confirm API key; check rate limits.
- **Audio Conversion Fails**: Install `pydub` dependencies (e.g., FFmpeg).
- **CORS Issues**: Ensure backend allows your frontend origin.

Logs are saved to `nova_ai.log`.


## Acknowledgements

- [OpenAI](https://openai.com) for AI models.
- [FastAPI](https://fastapi.tiangolo.com) for the backend framework.
- [Pydub](https://github.com/jiaaro/pydub) for audio processing.

For questions, open an issue or contact [your.email@example.com](mailto:your.email@example.com).
