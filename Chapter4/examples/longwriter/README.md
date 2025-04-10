# 长文生成器 (Long Writer)

使用OpenRouter AI的DeepSeek V3模型生成结构化的长篇文章，支持大纲生成、分章节内容生成和自动组装。

## 主要功能

- 自动生成文章大纲(包含引言和结论)
- 分章节生成内容，保持上下文一致性
- 处理API错误和内容审核
- 自动保存生成进度
- 输出Markdown格式文章

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 设置OpenRouter API密钥(任选一种方式):
   - 环境变量: `export OPENROUTER_API_KEY='your-api-key'`
   - 命令行参数: `--api_key your-api-key`

2. 运行脚本:
```bash
python main.py --topic "人工智能" --style "科普" --length 5000
```

## 参数说明

- `--topic`: 文章主题 (必填)
- `--style`: 文章风格 (如: 科普、正式、轻松等) (必填) 
- `--length`: 文章长度(字数) (必填)
- `--api_key`: OpenRouter API密钥 (可选，可通过环境变量设置)

## 输出文件

生成的文章会保存为`{主题}_article.md`文件，同时生成日志文件`longwriter.log`

## 示例

生成一篇5000字的科普风格AI文章:
```bash
python main.py --topic "人工智能" --style "科普" --length 5000
```

## 高级功能

- 支持自定义大纲模板(通过修改代码实现)
- 自动处理API速率限制
- 内容净化处理
- 详细的日志记录