# DeepSeek Writer

A Python program for long text generation using DeepSeek V3 model via OpenRouter API.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project directory with your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

Run the program:
```bash
python main.py
```

Enter your prompt when prompted, and the program will generate text using DeepSeek V3 model.

## Configuration

You can adjust generation parameters in `main.py`:
- `max_tokens`: Controls the length of generated text (default: 2000)
- `temperature`: Controls creativity/randomness (default: 0.7)

## Requirements

- Python 3.7+
- OpenRouter API key (free tier available)