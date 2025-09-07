# Introduction to the Long Writer Project

This document provides an overview of the "Long Writer" project found in the `Chapter4/examples/longwriter/` directory.

## Purpose

The Long Writer is a Python command-line tool designed to generate structured, long-form articles using AI. It leverages the DeepSeek V3 model via the OpenRouter API to create content based on a user-provided topic, style, and desired length.

## Key Features

Based on the `README.md` and `main.py` files, the key features include:

*   **Two-Stage Generation:**
    *   **Outline Generation:** Automatically creates a structured outline for the article, including an introduction and conclusion (`OutlineGenerator`).
    *   **Content Expansion:** Expands each section of the outline into detailed content, aiming for contextual consistency (`ContentExpander`).
*   **Structured Output:** Formats the final generated content into a Markdown file (`ArticleFormatter`).
*   **Configuration:** Accepts topic, style, and length as command-line arguments. API key can be provided via argument or environment variable (`OPENROUTER_API_KEY`).
*   **Error Handling & Logging:** Intended to handle API errors and maintains a log file (`longwriter.log`). (Note: The provided log indicates past generation failures).

## How it Works

1.  The user runs `main.py` with `--topic`, `--style`, and `--length` arguments.
2.  The `LongWriter` class orchestrates the process.
3.  `OutlineGenerator` is called to create an `ArticleOutline`.
4.  `ContentExpander` iterates through the outline sections, generating content for each.
5.  `ArticleFormatter` assembles the generated content into a final Markdown string.
6.  The result is printed to the console or saved to a file specified by the `--output` argument (defaults to `{topic}_article.md` according to the README).

## Dependencies

The project requires:

*   `openai>=1.0.0` (Likely used as the client library interface for OpenRouter)
*   `python-dotenv>=1.0.0` (For managing the API key via a `.env` file)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage Example

```bash
python main.py --topic "Artificial Intelligence" --style "Informative" --length 3000 --api_key your-openrouter-key
```
Or using an environment variable:
```bash
export OPENROUTER_API_KEY='your-openrouter-key'
python main.py --topic "Artificial Intelligence" --style "Informative" --length 3000
```

## Project Structure

*   `main.py`: Main script execution point.
*   `README.md`: Project description (in Chinese).
*   `requirements.txt`: Python dependencies.
*   `longwriter.log`: Log file for generation process.
*   `generators/`: Contains modules for outline (`outline.py`) and content (`content.py`) generation (contents not provided).
*   `schemas/`: Contains data structures, likely for the outline (`outline.py`) (contents not provided).
*   `utils/`: Contains utility modules, like the formatter (`formatter.py`) and potentially API interaction logic (contents not provided).