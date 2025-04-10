from typing import Optional, Dict, List
from openai import OpenAI

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='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',
            default_headers={
                "HTTP-Referer": "https://github.com/GenAI-From-Zero-to-Hero",
                "X-Title": "LongWriter"
            }
        )

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000
    ) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return None