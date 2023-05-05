# text-splitter

Large language models (LLMs) have lots of amazing use cases. But often they have a limited context size that is smaller than larger documents. To use documents of larger length, you often have to split your text into chunks to fit within this context size.

This crate attempts to maximize chunk size, while still preserving semantic units wherever possible.

## Get Started

### By Number of Characters

```
use text_splitter::{Characters, TextSplitter};

// Maximum number of characters in a chunk
let max_characters = 1000;
let splitter = TextSplitter::new(Characters::new(max_characters))
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunk_by_paragraphs("your document text");
```

### By Tokens

```
use text_splitter::{TextSplitter, Tokens};
// Can also use tiktoken-rs, or anything that implements the NumTokens
// trait from the text_splitter crate.
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
let max_tokens = 1000;
let splitter = TextSplitter::new(Tokens::new(tokenizer), 1000)
    // Optionally can also have the splitter trim whitespace for you
    .with_trim_chunks(true);

let chunks = splitter.chunk_by_paragraphs("your document text");
```

## Method

To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:

1. Split the text by a given level
2. For each section, does it fit within the chunk size?
   a. Yes. Fit as many of these neighboring sections into a chunk as possible.
   b. No. Split by the next level and repeat.

The boundaries used to split the text if using the top-level `chunk_by_paragraphs` method, in descending length:

1. 2 or more newlines (Newline is `\r\n`, `\n`, or `\r`)
2. 1 newline
3. [Unicode Sentences](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
4. [Unicode Words](https://www.unicode.org/reports/tr29/#Word_Boundaries)
5. [Unicode Graphemes](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
6. Characters

Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

_Note on sentences:_ There are lots of methods of determining sentence breaks, all to varying degrees of accuracy, and many requiring ML models to do so. Rather than trying to find the perfect sentence breaks, we rely on unicode method of sentence boundaries, which in most cases is good enough for finding a decent semantic breaking point if a paragraph is too large, and avoids the performance penalties of many other methods.

## Inspiration

This crate was inspired by [LangChain's TextSplitter](https://python.langchain.com/en/latest/modules/indexes/text_splitters/examples/recursive_text_splitter.html). But, looking into the implementation, there was potential for better performance as well as better semantic chunking.

A big thank you to the unicode-rs team for their [unicode-segmentation](https://crates.io/crates/unicode-segmentation) crate that manages a lot of the complexity of matching the Unicode rules for words and sentences.
