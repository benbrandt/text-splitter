# semantic-text-splitter

[![Documentation Status](https://readthedocs.org/projects/semantic-text-splitter/badge/?version=stable)](https://semantic-text-splitter.readthedocs.io/en/latest/?badge=latest) [![Licence](https://img.shields.io/crates/l/text-splitter)](https://github.com/benbrandt/text-splitter/blob/main/LICENSE.txt)

Large language models (LLMs) can be used for many tasks, but often have a limited context size that can be smaller than documents you might want to use. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate provides methods for splitting longer pieces of text into smaller chunks, aiming to maximize a desired chunk size, but still splitting at semantically sensible boundaries whenever possible.

## Get Started

### By Number of Characters

```python
from semantic_text_splitter import TextSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter not trim whitespace for you
splitter = TextSplitter(max_characters)
# splitter = TextSplitter(max_characters, trim=False)

chunks = splitter.chunks("your document text")
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```python
from semantic_text_splitter import TextSplitter


# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
splitter = TextSplitter((200,1000))

chunks = splitter.chunks("your document text")
```

### Using a Hugging Face Tokenizer

```python
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

# Maximum number of tokens in a chunk
max_tokens = 1000
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

chunks = splitter.chunks("your document text")
```

### Using a Tiktoken Tokenizer

```python
from semantic_text_splitter import TextSplitter

# Maximum number of tokens in a chunk
max_tokens = 1000
splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", max_tokens)

chunks = splitter.chunks("your document text")
```

### Using a Custom Callback

```python
from semantic_text_splitter import TextSplitter

splitter = TextSplitter.from_callback(lambda text: len(text), 1000)

chunks = splitter.chunks("your document text")
```

### Markdown

All of the above examples also can also work with Markdown text. You can use the `MarkdownSplitter` in the same ways as the `TextSplitter`.

```python
from semantic_text_splitter import MarkdownSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter not trim whitespace for you
splitter = MarkdownSplitter(max_characters)
# splitter = MarkdownSplitter(max_characters, trim=False)

chunks = splitter.chunks("# Header\n\nyour document text")
```

## Method

To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:

1. Split the text by a increasing semantic levels.
2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length. Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.

The boundaries used to split the text if using the `chunks` method, in ascending order:

### `TextSplitter` Semantic Levels

1. Characters
2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
5. Ascending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level. So a sequence of 2 newlines is a higher level than a sequence of 1 newline, and so on.

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

### `MarkdownSplitter` Semantic Levels

Markdown is parsed according to the `CommonMark` spec, along with some optional features such as GitHub Flavored Markdown.

1. Characters
2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
7. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
8. Thematic breaks or horizontal rules.
9. Headings by level

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

### Note on sentences

There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

## Inspiration

This crate was inspired by [LangChain's TextSplitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html#langchain_text_splitters.character.RecursiveCharacterTextSplitter). But, looking into the implementation, there was potential for better performance as well as better semantic chunking.

A big thank you to the unicode-rs team for their [unicode-segmentation](https://crates.io/crates/unicode-segmentation) crate that manages a lot of the complexity of matching the Unicode rules for words and sentences.
