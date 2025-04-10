from longwriter.schemas.outline import ArticleOutline

class ArticleFormatter:
    @staticmethod
    def format_article(title: str, sections: list) -> str:
        formatted = f"# {title}\n\n"
        for section in sections:
            formatted += f"## {section['title']}\n\n"
            formatted += f"{section['content']}\n\n"
        return formatted

    @staticmethod
    def save_to_file(content: str, filename: str) -> bool:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False