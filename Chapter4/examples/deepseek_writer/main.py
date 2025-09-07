import requests
import json
from config import Config

class DeepSeekWriter:
    def __init__(self):
        Config.validate()
        self.headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Optional
            "X-Title": "DeepSeek Writer"  # Optional
        }

    def generate_text(self, prompt, max_tokens=2000, temperature=0.7):
        """Generate long text using DeepSeek V3 model"""
        payload = {
            "model": Config.DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                Config.OPENROUTER_API_URL,
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error generating text: {e}")
            return None

if __name__ == "__main__":
    writer = DeepSeekWriter()
    print("DeepSeek Long Text Generator")
    print("Enter your prompt (type 'exit' to quit):")
    
    while True:
        prompt = input("> ")
        if prompt.lower() == 'exit':
            break
            
        print("\nGenerating...\n")
        result = writer.generate_text(prompt)
        if result:
            print(f"\nGenerated Text:\n{result}\n")
        else:
            print("Failed to generate text. Please try again.")