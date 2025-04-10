from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Section:
    title: str
    description: str
    word_count: int

@dataclass
class ArticleOutline:
    title: str
    style: str
    total_words: int
    sections: List[Section]