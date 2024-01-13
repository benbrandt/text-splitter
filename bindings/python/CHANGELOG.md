# Changelog

## v0.5.1 and beyond

See main CHANGELOG.md for the repo.

## v0.3.1

Fix broken Github release

## v0.3.0

### What's New

- Update to `v0.5.0` of `text-splitter` for significant performance improvements for generating chunks with the `tokenizers` or `tiktoken-rs` crates by applying binary search when attempting to find the next matching chunk size.

### Breaking Changes

- Minimum Python version is now 3.8.
- Due to using binary search, there are some slight differences at the edges of chunks where the algorithm was a little greedier before. If two candidates would tokenize to the same amount of tokens that fit within the capacity, it will now choose the shorter text. Due to the nature of of tokenizers, this happens more often with whitespace at the end of a chunk, and rarely effects users who have set `trim_chunks=true`. It is a tradeoff, but would have made the binary search code much more complicated to keep the exact same behavior.

## v0.2.4

### What's New

- Update to v0.4.5 of `text-splitter` to support `tokenizers@0.15.0`
- Update `tokenizers` and `tiktoken-rs` to latest version

## v0.2.3

### What's New

- Update to v0.4.4 of `text-splitter` to support `tokenizers@0.14.0`
- Update `tokenizers` and `tiktoken-rs` to latest versions

## v0.2.2

### What's New

- Update to v0.4.2 of `text-splitter` to support `tiktoken-rs@0.5.0`

## v0.2.1

### What's New

- Support Open AI Tiktoken tokenizers. So you can now give an OpenAI model name to tokenize the text for when calculating chunk sizes.

```python
from semantic_text_splitter import TiktokenTextSplitter

# Maximum number of tokens in a chunk
max_tokens = 1000
# Optionally can also have the splitter not trim whitespace for you
splitter = TiktokenTextSplitter("gpt-3.5-turbo", trim_chunks=False)

chunks = splitter.chunks("your document text", max_tokens)
```

## v0.2.0

### What's New

- New `HuggingFaceTextSplitter`, which allows for using Hugging Face's `tokenizers` package to count chunks by tokens with a tokenizer of your choice.

```python
from semantic_text_splitter import HuggingFaceTextSplitter
from tokenizers import Tokenizer

# Maximum number of tokens in a chunk
max_characters = 1000
# Optionally can also have the splitter not trim whitespace for you
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)

chunks = splitter.chunks("your document text", max_characters)
```

### Breaking Changes

- `trim_chunks` now defaults to `True` instead of `False`. For most use cases, this is the desired behavior, especially with chunk ranges.

## v0.1.4

Fifth time is the charm?

## v0.1.3

Rename package to `semantic-text-splitter` so it can actually be uploaded to PyPi.

## v0.1.2

Fix bad release

## v0.1.1

Fix bad release

## v0.1.0

Initial release with just `CharacterTextSplitter`
