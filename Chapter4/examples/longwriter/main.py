import argparse
import os
from typing import Optional
from longwriter.generators.outline import OutlineGenerator
from longwriter.generators.content import ContentExpander
from longwriter.utils.formatter import ArticleFormatter
from longwriter.schemas.outline import ArticleOutline, Section

class LongWriter:
    def __init__(self, api_key: str):
        self.outline_gen = OutlineGenerator(api_key)
        self.content_exp = ContentExpander(api_key)
        
    def generate_article(self, topic: str, style: str, length: int) -> Optional[str]:
        # 第一阶段：生成大纲
        outline = self.outline_gen.generate_outline(topic, style, length)
        if not outline:
            return None

        # 第二阶段：扩展内容
        sections = []
        for section in outline.sections:
            content = self.content_exp.expand_section(outline, section)
            if content:
                sections.append({
                    'title': section.title,
                    'content': content
                })

        # 格式化最终文章
        return ArticleFormatter.format_article(outline.title, sections)

def main():
    parser = argparse.ArgumentParser(description="两阶段长文生成器")
    parser.add_argument("--topic", required=True, help="文章主题")
    parser.add_argument("--style", required=True, help="文章风格")
    parser.add_argument("--length", type=int, required=True, help="文章长度(字数)")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--api_key", help="OpenRouter API密钥，或设置环境变量OPENROUTER_API_KEY")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("错误: 必须提供OpenRouter API密钥")
        return

    writer = LongWriter(api_key)
    article = writer.generate_article(args.topic, args.style, args.length)

    if article:
        if args.output:
            ArticleFormatter.save_to_file(article, args.output)
            print(f"文章已保存到 {args.output}")
        else:
            print("\n生成的文章:\n")
            print(article)
    else:
        print("文章生成失败")

if __name__ == "__main__":
    main()
