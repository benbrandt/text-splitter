//! Python Bindings for text-splitter crate

#![warn(
    clippy::pedantic,
    future_incompatible,
    missing_debug_implementations,
    missing_docs,
    nonstandard_style,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    unused
)]
// pyo3 uses this
#![allow(elided_lifetimes_in_paths)]

use std::str::FromStr;

use auto_enums::auto_enum;
use pyo3::{exceptions::PyException, prelude::*, pybacked::PyBackedStr};
use text_splitter::{
    Characters, ChunkCapacity, ChunkSize, ChunkSizer, MarkdownSplitter, TextSplitter,
};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tokenizers::Tokenizer;

/// Custom chunk capacity for python to make it easier to work
/// with python arguments
#[derive(Debug, FromPyObject)]
enum PyChunkCapacity {
    #[pyo3(transparent, annotation = "int")]
    Int(usize),
    #[pyo3(annotation = "tuple[int, int]")]
    IntTuple(usize, usize),
}

impl ChunkCapacity for PyChunkCapacity {
    fn start(&self) -> Option<usize> {
        match self {
            Self::Int(_) => None,
            Self::IntTuple(start, _) => Some(*start),
        }
    }

    fn end(&self) -> usize {
        match self {
            Self::Int(end) | Self::IntTuple(_, end) => *end,
        }
    }
}

/// Newtype around a Python callback so we can `impl ChunkSizer`
struct CustomCallback(PyObject);

impl ChunkSizer for CustomCallback {
    /// Determine the size of a given chunk to use for validation
    fn chunk_size(&self, chunk: &str, capacity: &impl ChunkCapacity) -> ChunkSize {
        Python::with_gil(|py| {
            let size = self
                .0
                .call_bound(py, (chunk,), None)
                .unwrap()
                .extract::<usize>(py)
                .unwrap();

            ChunkSize::from_size(size, capacity)
        })
    }
}

/// Keeps track of the corresponding byte to character offset in a text
struct ByteToCharOffsetTracker<'text> {
    byte_offset: usize,
    char_offset: usize,
    text: &'text str,
}

impl<'text> ByteToCharOffsetTracker<'text> {
    fn new(text: &'text str) -> Self {
        Self {
            byte_offset: 0,
            char_offset: 0,
            text,
        }
    }

    /// Updates the current offsets, but is able to cache previous results
    fn map_byte_to_char(&mut self, (offset, chunk): (usize, &'text str)) -> (usize, &'text str) {
        let prev_text = self
            .text
            .get(self.byte_offset..offset)
            .expect("Invalid byte sequence");
        self.byte_offset = offset;
        self.char_offset += prev_text.chars().count();
        (self.char_offset, chunk)
    }
}

#[allow(clippy::large_enum_variant)]
enum TextSplitterOptions {
    Characters(TextSplitter<Characters>),
    CustomCallback(TextSplitter<CustomCallback>),
    Huggingface(TextSplitter<Tokenizer>),
    Tiktoken(TextSplitter<CoreBPE>),
}

impl TextSplitterOptions {
    #[auto_enum(Iterator)]
    fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        match self {
            Self::Characters(splitter) => splitter.chunks(text, chunk_capacity),
            Self::CustomCallback(splitter) => splitter.chunks(text, chunk_capacity),
            Self::Huggingface(splitter) => splitter.chunks(text, chunk_capacity),
            Self::Tiktoken(splitter) => splitter.chunks(text, chunk_capacity),
        }
    }

    #[auto_enum(Iterator)]
    fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        match self {
            Self::Characters(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::CustomCallback(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::Huggingface(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::Tiktoken(splitter) => splitter.chunk_indices(text, chunk_capacity),
        }
    }
}

/**
Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

### By Number of Characters

```python
from semantic_text_splitter import TextSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter not trim whitespace for you
splitter = TextSplitter()
# splitter = TextSplitter(trim_chunks=False)

chunks = splitter.chunks("your document text", max_characters)
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```python
from semantic_text_splitter import TextSplitter

splitter = TextSplitter()

# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
```

### Using a Hugging Face Tokenizer

```python
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer

# Maximum number of tokens in a chunk
max_tokens = 1000
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)

chunks = splitter.chunks("your document text", max_tokens)
```

### Using a Tiktoken Tokenizer


```python
from semantic_text_splitter import TextSplitter

# Maximum number of tokens in a chunk
max_tokens = 1000
splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo")

chunks = splitter.chunks("your document text", max_tokens)
```

### Using a Custom Callback

```python
from semantic_text_splitter import TextSplitter

# Optionally can also have the splitter trim whitespace for you
splitter = TextSplitter.from_callback(lambda text: len(text))

# Maximum number of tokens in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
```

Args:
    trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
        beginning and end or not. If False, joining all chunks will return the original
        string. Defaults to True.
*/
#[pyclass(frozen, name = "TextSplitter")]
struct PyTextSplitter {
    splitter: TextSplitterOptions,
}

#[pymethods]
impl PyTextSplitter {
    #[new]
    #[pyo3(signature = (trim_chunks=true))]
    fn new(trim_chunks: bool) -> Self {
        Self {
            splitter: TextSplitterOptions::Characters(
                TextSplitter::default().with_trim_chunks(trim_chunks),
            ),
        }
    }

    /**
    Instantiate a new text splitter from a Hugging Face Tokenizer instance.

    Args:
        tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
            chunk.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

    Returns:
        The new text splitter
    */
    #[staticmethod]
    #[pyo3(signature = (tokenizer, trim_chunks=true))]
    fn from_huggingface_tokenizer(
        tokenizer: &Bound<'_, PyAny>,
        trim_chunks: bool,
    ) -> PyResult<Self> {
        // Get the json out so we can reconstruct the tokenizer on the Rust side
        let json = tokenizer.call_method0("to_str")?.extract::<PyBackedStr>()?;
        let tokenizer =
            Tokenizer::from_str(&json).map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: TextSplitterOptions::Huggingface(
                TextSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new text splitter from the given Hugging Face Tokenizer JSON string.

    Args:
        json (str): A valid JSON string representing a previously serialized
            Hugging Face Tokenizer
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.

    Returns:
        The new text splitter
    */
    #[staticmethod]
    #[pyo3(signature = (json, trim_chunks=true))]
    fn from_huggingface_tokenizer_str(json: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer = json
            .parse()
            .map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: TextSplitterOptions::Huggingface(
                TextSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new text splitter from the Hugging Face tokenizer file at the given path.

    Args:
        path (str): A path to a local JSON file representing a previously serialized
            Hugging Face tokenizer.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.

    Returns:
        The new text splitter
    */
    #[staticmethod]
    #[pyo3(signature = (path, trim_chunks=true))]
    fn from_huggingface_tokenizer_file(path: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| PyException::new_err(format!("{e}")))?;
        Ok(Self {
            splitter: TextSplitterOptions::Huggingface(
                TextSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new text splitter based on an OpenAI Tiktoken tokenizer.

    Args:
        model (str): The OpenAI model name you want to retrieve a tokenizer for.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.

    Returns:
        The new text splitter
    */
    #[staticmethod]
    #[pyo3(signature = (model, trim_chunks=true))]
    fn from_tiktoken_model(model: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer =
            get_bpe_from_model(model).map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: TextSplitterOptions::Tiktoken(
                TextSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new text splitter based on a custom callback.

    Args:
        callback (Callable[[str], int]): A lambda or other function that can be called. It will be
            provided a piece of text, and it should return an integer value for the size.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.

    Returns:
        The new text splitter
    */
    #[staticmethod]
    #[pyo3(signature = (callback, trim_chunks=true))]
    fn from_callback(callback: PyObject, trim_chunks: bool) -> Self {
        Self {
            splitter: TextSplitterOptions::CustomCallback(
                TextSplitter::new(CustomCallback(callback)).with_trim_chunks(trim_chunks),
            ),
        }
    }

    /**
    Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.


    ## Method

    To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:

    1. Split the text by a increasing semantic levels.
    2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
    3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length. Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.

    The boundaries used to split the text if using the `chunks` method, in ascending order:

    1. Characters
    2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    5. Ascending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level. So a sequence of 2 newlines is a higher level than a sequence of 1 newline, and so on.

    Args:
        text (str): Text to split.
        chunk_capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.

    Returns:
        A list of strings, one for each chunk. If `trim_chunks` was specified in the text
        splitter, then each chunk will already be trimmed as well.
    */
    fn chunks<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> Vec<&'text str> {
        self.splitter.chunks(text, chunk_capacity).collect()
    }

    /**
    Generate a list of chunks from a given text, along with their character offsets in the original text. Each chunk will be up to the `chunk_capacity`.

    See `chunks` for more information.

    Args:
        text (str): Text to split.
        chunk_capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.

    Returns:
        A list of tuples, one for each chunk. The first item will be the character offset relative
        to the original text. The second item is the chunk itself.
        If `trim_chunks` was specified in the text splitter, then each chunk will already be
        trimmed as well.
    */
    fn chunk_indices<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> Vec<(usize, &'text str)> {
        let mut offsets = ByteToCharOffsetTracker::new(text);
        self.splitter
            .chunk_indices(text, chunk_capacity)
            .map(|c| offsets.map_byte_to_char(c))
            .collect()
    }
}

#[allow(clippy::large_enum_variant)]
enum MarkdownSplitterOptions {
    Characters(MarkdownSplitter<Characters>),
    CustomCallback(MarkdownSplitter<CustomCallback>),
    Huggingface(MarkdownSplitter<Tokenizer>),
    Tiktoken(MarkdownSplitter<CoreBPE>),
}

impl MarkdownSplitterOptions {
    #[auto_enum(Iterator)]
    fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        match self {
            Self::Characters(splitter) => splitter.chunks(text, chunk_capacity),
            Self::CustomCallback(splitter) => splitter.chunks(text, chunk_capacity),
            Self::Huggingface(splitter) => splitter.chunks(text, chunk_capacity),
            Self::Tiktoken(splitter) => splitter.chunks(text, chunk_capacity),
        }
    }

    #[auto_enum(Iterator)]
    fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        match self {
            Self::Characters(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::CustomCallback(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::Huggingface(splitter) => splitter.chunk_indices(text, chunk_capacity),
            Self::Tiktoken(splitter) => splitter.chunk_indices(text, chunk_capacity),
        }
    }
}

/**
Markdown splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

### By Number of Characters

```python
from semantic_text_splitter import MarkdownSplitter

# Maximum number of characters in a chunk
max_characters = 1000
# Optionally can also have the splitter not trim whitespace for you
splitter = MarkdownSplitter()
# splitter = MarkdownSplitter(trim_chunks=False)

chunks = splitter.chunks("# Header\n\nyour document text", max_characters)
```

### Using a Range for Chunk Capacity

You also have the option of specifying your chunk capacity as a range.

Once a chunk has reached a length that falls within the range it will be returned.

It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

```python
from semantic_text_splitter import MarkdownSplitter

splitter = MarkdownSplitter()

# Maximum number of characters in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks("# Header\n\nyour document text", chunk_capacity=(200,1000))
```

### Using a Hugging Face Tokenizer

```python
from semantic_text_splitter import MarkdownSplitter
from tokenizers import Tokenizer

# Maximum number of tokens in a chunk
max_tokens = 1000
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer)

chunks = splitter.chunks("# Header\n\nyour document text", max_tokens)
```

### Using a Tiktoken Tokenizer


```python
from semantic_text_splitter import MarkdownSplitter

# Maximum number of tokens in a chunk
max_tokens = 1000
splitter = MarkdownSplitter.from_tiktoken_model("gpt-3.5-turbo")

chunks = splitter.chunks("# Header\n\nyour document text", max_tokens)
```

### Using a Custom Callback

```python
from semantic_text_splitter import MarkdownSplitter

# Optionally can also have the splitter trim whitespace for you
splitter = MarkdownSplitter.from_callback(lambda text: len(text))

# Maximum number of tokens in a chunk. Will fill up the
# chunk until it is somewhere in this range.
chunks = splitter.chunks("# Header\n\nyour document text", chunk_capacity=(200,1000))
```

Args:
    trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
        beginning and end or not. If False, joining all chunks will return the original
        string. Indentation however will be preserved if the chunk also includes multiple lines.
        Extra newlines are always removed, but if the text would include multiple indented list
        items, the indentation of the first element will also be preserved.
        Defaults to True.
*/
#[pyclass(frozen, name = "MarkdownSplitter")]
struct PyMarkdownSplitter {
    splitter: MarkdownSplitterOptions,
}

#[pymethods]
impl PyMarkdownSplitter {
    #[new]
    #[pyo3(signature = (trim_chunks=true))]
    fn new(trim_chunks: bool) -> Self {
        Self {
            splitter: MarkdownSplitterOptions::Characters(
                MarkdownSplitter::default().with_trim_chunks(trim_chunks),
            ),
        }
    }

    /**
    Instantiate a new markdown splitter from a Hugging Face Tokenizer instance.

    Args:
        tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
            chunk.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Indentation however will be preserved if the chunk also includes multiple lines.
            Extra newlines are always removed, but if the text would include multiple indented list
            items, the indentation of the first element will also be preserved.
            Defaults to True.

    Returns:
        The new markdown splitter
    */
    #[staticmethod]
    #[pyo3(signature = (tokenizer, trim_chunks=true))]
    fn from_huggingface_tokenizer(
        tokenizer: &Bound<'_, PyAny>,
        trim_chunks: bool,
    ) -> PyResult<Self> {
        // Get the json out so we can reconstruct the tokenizer on the Rust side
        let json = tokenizer.call_method0("to_str")?.extract::<PyBackedStr>()?;
        let tokenizer =
            Tokenizer::from_str(&json).map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: MarkdownSplitterOptions::Huggingface(
                MarkdownSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new markdown splitter from the given Hugging Face Tokenizer JSON string.

    Args:
        json (str): A valid JSON string representing a previously serialized
            Hugging Face Tokenizer
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Indentation however will be preserved if the chunk also includes multiple lines.
            Extra newlines are always removed, but if the text would include multiple indented list
            items, the indentation of the first element will also be preserved.
            Defaults to True.

    Returns:
        The new markdown splitter
    */
    #[staticmethod]
    #[pyo3(signature = (json, trim_chunks=true))]
    fn from_huggingface_tokenizer_str(json: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer = json
            .parse()
            .map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: MarkdownSplitterOptions::Huggingface(
                MarkdownSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new markdown splitter from the Hugging Face tokenizer file at the given path.

    Args:
        path (str): A path to a local JSON file representing a previously serialized
            Hugging Face tokenizer.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Indentation however will be preserved if the chunk also includes multiple lines.
            Extra newlines are always removed, but if the text would include multiple indented list
            items, the indentation of the first element will also be preserved.
            Defaults to True.

    Returns:
        The new markdown splitter
    */
    #[staticmethod]
    #[pyo3(signature = (path, trim_chunks=true))]
    fn from_huggingface_tokenizer_file(path: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| PyException::new_err(format!("{e}")))?;
        Ok(Self {
            splitter: MarkdownSplitterOptions::Huggingface(
                MarkdownSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a new markdown splitter based on an OpenAI Tiktoken tokenizer.

    Args:
        model (str): The OpenAI model name you want to retrieve a tokenizer for.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Indentation however will be preserved if the chunk also includes multiple lines.
            Extra newlines are always removed, but if the text would include multiple indented list
            items, the indentation of the first element will also be preserved.
            Defaults to True.

    Returns:
        The new markdown splitter
    */
    #[staticmethod]
    #[pyo3(signature = (model, trim_chunks=true))]
    fn from_tiktoken_model(model: &str, trim_chunks: bool) -> PyResult<Self> {
        let tokenizer =
            get_bpe_from_model(model).map_err(|e| PyException::new_err(format!("{e}")))?;

        Ok(Self {
            splitter: MarkdownSplitterOptions::Tiktoken(
                MarkdownSplitter::new(tokenizer).with_trim_chunks(trim_chunks),
            ),
        })
    }

    /**
    Instantiate a markdown text splitter based on a custom callback.

    Args:
        callback (Callable[[str], int]): A lambda or other function that can be called. It will be
            provided a piece of text, and it should return an integer value for the size.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Indentation however will be preserved if the chunk also includes multiple lines.
            Extra newlines are always removed, but if the text would include multiple indented list
            items, the indentation of the first element will also be preserved.
            Defaults to True.

    Returns:
        The new markdown splitter
    */
    #[staticmethod]
    #[pyo3(signature = (callback, trim_chunks=true))]
    fn from_callback(callback: PyObject, trim_chunks: bool) -> Self {
        Self {
            splitter: MarkdownSplitterOptions::CustomCallback(
                MarkdownSplitter::new(CustomCallback(callback)).with_trim_chunks(trim_chunks),
            ),
        }
    }

    /**
    Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

    ## Method

    To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:

    1. Split the text by a increasing semantic levels.
    2. Check the first item for each level and select the highest level whose first item still fits within the chunk size.
    3. Merge as many of these neighboring sections of this level or above into a chunk to maximize chunk length. Boundaries of higher semantic levels are always included when merging, so that the chunk doesn't inadvertantly cross semantic boundaries.

    The boundaries used to split the text if using the `chunks` method, in ascending order:

    1. Characters
    2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
    6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
    7. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
    8. Thematic breaks or horizontal rules.
    9. Headings by level

    Markdown is parsed according to the Commonmark spec, along with some optional features such as GitHub Flavored Markdown.

    Args:
        text (str): Text to split.
        chunk_capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.

    Returns:
        A list of strings, one for each chunk. If `trim_chunks` was specified in the text
        splitter, then each chunk will already be trimmed as well.
    */
    fn chunks<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> Vec<&'text str> {
        self.splitter.chunks(text, chunk_capacity).collect()
    }

    /**
    Generate a list of chunks from a given text, along with their character offsets in the original text. Each chunk will be up to the `chunk_capacity`.

    See `chunks` for more information.

    Args:
        text (str): Text to split.
        chunk_capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.

    Returns:
        A list of tuples, one for each chunk. The first item will be the character offset relative
        to the original text. The second item is the chunk itself.
        If `trim_chunks` was specified in the text splitter, then each chunk will already be
        trimmed as well.
    */
    fn chunk_indices<'text, 'splitter: 'text>(
        &'splitter self,
        text: &'text str,
        chunk_capacity: PyChunkCapacity,
    ) -> Vec<(usize, &'text str)> {
        let mut offsets = ByteToCharOffsetTracker::new(text);
        self.splitter
            .chunk_indices(text, chunk_capacity)
            .map(|c| offsets.map_byte_to_char(c))
            .collect()
    }
}

#[doc = include_str!("../README.md")]
#[pymodule]
fn semantic_text_splitter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTextSplitter>()?;
    m.add_class::<PyMarkdownSplitter>()?;
    Ok(())
}
