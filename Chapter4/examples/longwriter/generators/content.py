from typing import Optional
from longwriter.schemas.outline import ArticleOutline, Section
from longwriter.utils.api import OpenRouterClient

class ContentExpander:
    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key)
        
    def expand_section(self, outline: ArticleOutline, section: Section) -> Optional[str]:
        prompt = f"""根据以下要求扩展文章章节：
- 文章标题：{outline.title}
- 写作风格：{outline.style}
- 章节标题：{section.title}
- 章节描述：{section.description}
- 目标字数：{section.word_count}字
"""
        try:
            response = self.client.chat_completion(
                model="deepseek/deepseek-chat-v3-0324:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=section.word_count * 2
            )
            return response
        except Exception as e:
            print(f"章节扩展失败: {e}")
            return None