from typing import Optional
from longwriter.schemas.outline import ArticleOutline, Section
from longwriter.utils.api import OpenRouterClient

class OutlineGenerator:
    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key)
        
    def generate_outline(self, topic: str, style: str, length: int) -> Optional[ArticleOutline]:
        prompt = f"""请为{topic}生成一个详细的大纲，要求：
- 写作风格：{style}
- 总字数：{length}字
- 返回格式：标题和段落描述
"""
        try:
            response = self.client.chat_completion(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"大纲生成失败: {e}")
            return None

    def _parse_response(self, response: str) -> ArticleOutline:
        # 实现解析逻辑
        pass