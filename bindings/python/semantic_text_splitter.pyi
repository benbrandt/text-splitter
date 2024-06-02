from typing import Any, Callable, List, Tuple, Union, final


@final
class TextSplitter:
    """Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

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

    Args:
        capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.
        overlap (int, optional): The maximum number of allowed characters to overlap between chunks.
            Defaults to 0.
        trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(
        self, capacity: Union[int, Tuple[int, int]], overlap: int = 0, trim: bool = True
    ) -> None: ...

    @staticmethod
    def from_huggingface_tokenizer(
        tokenizer,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> TextSplitter:
        """Instantiate a new text splitter from a Hugging Face Tokenizer instance.

        Args:
            tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
                chunk.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_str(
        json: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> TextSplitter:
        """Instantiate a new text splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_file(
        path: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> TextSplitter:
        """Instantiate a new text splitter from the Hugging Face tokenizer file at the given path.

        Args:
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_tiktoken_model(
        model: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> TextSplitter:
        """Instantiate a new text splitter based on an OpenAI Tiktoken tokenizer.

        Args:
            model (str): The OpenAI model name you want to retrieve a tokenizer for.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_callback(
        callback: Callable[[str], int],
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> TextSplitter:
        """Instantiate a new text splitter based on a custom callback.

        Args:
            callback (Callable[[str], int]): A lambda or other function that can be called. It will be
                provided a piece of text, and it should return an integer value for the size.
            capacity (int | (int, int)): The capacity of each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum allowed overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    def chunks(self, text: str) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `capacity`.


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

        Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.

        Args:
            text (str): Text to split.

        Returns:
            A list of strings, one for each chunk. If `trim` was specified in the text
            splitter, then each chunk will already be trimmed as well.
        """

    def chunk_indices(self, text: str) -> List[Tuple[int, str]]:
        """Generate a list of chunks from a given text, along with their character offsets in the original text. Each chunk will be up to the `capacity`.

        See `chunks` for more information.

        Args:
            text (str): Text to split.

        Returns:
            A list of tuples, one for each chunk. The first item will be the character offset relative
            to the original text. The second item is the chunk itself.
            If `trim` was specified in the text splitter, then each chunk will already be
            trimmed as well.
        """


@final
class MarkdownSplitter:
    """Markdown splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    ### By Number of Characters

    ```python
    from semantic_text_splitter import MarkdownSplitter

    # Maximum number of characters in a chunk
    max_characters = 1000
    # Optionally can also have the splitter not trim whitespace for you
    splitter = MarkdownSplitter(max_characters)
    # splitter = MarkdownSplitter(max_characters, trim=False)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Range for Chunk Capacity

    You also have the option of specifying your chunk capacity as a range.

    Once a chunk has reached a length that falls within the range it will be returned.

    It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

    ```python
    from semantic_text_splitter import MarkdownSplitter

    splitter = MarkdownSplitter(capacity=(200,1000))

    # Maximum number of characters in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Hugging Face Tokenizer

    ```python
    from semantic_text_splitter import MarkdownSplitter
    from tokenizers import Tokenizer

    # Maximum number of tokens in a chunk
    max_tokens = 1000
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Tiktoken Tokenizer


    ```python
    from semantic_text_splitter import MarkdownSplitter

    # Maximum number of tokens in a chunk
    max_tokens = 1000
    splitter = MarkdownSplitter.from_tiktoken_model("gpt-3.5-turbo"m max_tokens)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Custom Callback

    ```python
    from semantic_text_splitter import MarkdownSplitter

    # Optionally can also have the splitter trim whitespace for you
    splitter = MarkdownSplitter.from_callback(lambda text: len(text), 1000)

    # Maximum number of tokens in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    Args:
        capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.
        overlap (int, optional): The maximum number of allowed characters to overlap between chunks.
            Defaults to 0.
        trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(
        self, capacity: Union[int, Tuple[int, int]], overlap: int = 0, trim: bool = True
    ) -> None: ...

    @staticmethod
    def from_huggingface_tokenizer(
        tokenizer,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from a Hugging Face Tokenizer instance.

        Args:
            tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
                chunk.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_str(
        json: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_file(
        path: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from the Hugging Face tokenizer file at the given path.

        Args:
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_tiktoken_model(
        model: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter based on an OpenAI Tiktoken tokenizer.

        Args:
            model (str): The OpenAI model name you want to retrieve a tokenizer for.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_callback(
        callback: Callable[[str], int],
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter based on a custom callback.

        Args:
            callback (Callable[[str], int]): A lambda or other function that can be called. It will be
                provided a piece of text, and it should return an integer value for the size.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    def chunks(self, text: str) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `capacity`.

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

        Returns:
            A list of strings, one for each chunk. If `trim` was specified in the text
            splitter, then each chunk will already be trimmed as well.
        """

    def chunk_indices(self, text: str) -> List[Tuple[int, str]]:
        """Generate a list of chunks from a given text, along with their character offsets in the original text. Each chunk will be up to the `capacity`.

        See `chunks` for more information.

        Args:
            text (str): Text to split.

        Returns:
            A list of tuples, one for each chunk. The first item will be the character offset relative
            to the original text. The second item is the chunk itself.
            If `trim` was specified in the text splitter, then each chunk will already be
            trimmed as well.
        """


@final
class CodeSplitter:
    """Code splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    Uses [tree-sitter grammars](https://tree-sitter.github.io/tree-sitter/#parsers) for parsing the code.

    ### By Number of Characters

    ```python
    from semantic_text_splitter import CodeSplitter
    # Import the tree-sitter grammar you want to use
    import tree_sitter_python

    # Maximum number of characters in a chunk
    max_characters = 1000
    splitter = CodeSplitter(tree_sitter_python.language(), max_characters)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Range for Chunk Capacity

    You also have the option of specifying your chunk capacity as a range.

    Once a chunk has reached a length that falls within the range it will be returned.

    It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

    ```python
    from semantic_text_splitter import CodeSplitter
    # Import the tree-sitter grammar you want to use
    import tree_sitter_python

    splitter = CodeSplitter(tree_sitter_python.language(), (200,1000))

    # Maximum number of characters in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Hugging Face Tokenizer

    ```python
    from semantic_text_splitter import CodeSplitter
    from tokenizers import Tokenizer
    # Import the tree-sitter grammar you want to use
    import tree_sitter_python

    # Maximum number of tokens in a chunk
    max_tokens = 1000
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = CodeSplitter.from_huggingface_tokenizer(tree_sitter_python.language(), tokenizer, max_tokens)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Tiktoken Tokenizer


    ```python
    from semantic_text_splitter import CodeSplitter
    # Import the tree-sitter grammar you want to use
    import tree_sitter_python

    # Maximum number of tokens in a chunk
    max_tokens = 1000
    splitter = CodeSplitter.from_tiktoken_model(tree_sitter_python.language(), "gpt-3.5-turbo", max_tokens)

    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    ### Using a Custom Callback

    ```python
    from semantic_text_splitter import CodeSplitter
    # Import the tree-sitter grammar you want to use
    import tree_sitter_python

    # Optionally can also have the splitter trim whitespace for you
    splitter = CodeSplitter.from_callback(tree_sitter_python.language(), lambda text: len(text), (200,1000))

    # Maximum number of tokens in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("# Header\n\nyour document text")
    ```

    Args:
        language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
            to use for parsing the code.
        capacity (int | (int, int)): The capacity of characters in each chunk. If a
            single int, then chunks will be filled up as much as possible, without going over
            that number. If a tuple of two integers is provided, a chunk will be considered
            "full" once it is within the two numbers (inclusive range). So it will only fill
            up the chunk until the lower range is met.
        overlap (int, optional): The maximum number of allowed characters to overlap between chunks.
            Defaults to 0.
        trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(
        self,
        language: int,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> None: ...

    @staticmethod
    def from_huggingface_tokenizer(
        language: int,
        tokenizer,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new code splitter from a Hugging Face Tokenizer instance.

        Args:
            language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
                to use for parsing the code.
            tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
                chunk.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new code splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_str(
        language: int,
        json: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new code splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
                to use for parsing the code.
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new code splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_file(
        language: int,
        path: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new code splitter from the Hugging Face tokenizer file at the given path.

        Args:
            language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
                to use for parsing the code.
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new code splitter
        """

    @staticmethod
    def from_tiktoken_model(
        language: int,
        model: str,
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a new code splitter based on an OpenAI Tiktoken tokenizer.

        Args:
            language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
                to use for parsing the code.
            model (str): The OpenAI model name you want to retrieve a tokenizer for.
            capacity (int | (int, int)): The capacity of tokens in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum number of allowed tokens to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new code splitter
        """

    @staticmethod
    def from_callback(
        language: int,
        callback: Callable[[str], int],
        capacity: Union[int, Tuple[int, int]],
        overlap: int = 0,
        trim: bool = True,
    ) -> MarkdownSplitter:
        """Instantiate a code text splitter based on a custom callback.

        Args:
            language (int): The [tree-sitter language](https://tree-sitter.github.io/tree-sitter/#parsers)
                to use for parsing the code.
            callback (Callable[[str], int]): A lambda or other function that can be called. It will be
                provided a piece of text, and it should return an integer value for the size.
            capacity (int | (int, int)): The capacity of each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.
            overlap (int, optional): The maximum allowed overlap to overlap between chunks.
                Defaults to 0.
            trim (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new code splitter
        """

    def chunks(self, text: str) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `capacity`.

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
        5. Ascending depth of the syntax tree. So function would have a higher level than a statement inside of the function, and so on.

        Args:
            text (str): Text to split.

        Returns:
            A list of strings, one for each chunk. If `trim` was specified in the text
            splitter, then each chunk will already be trimmed as well.
        """

    def chunk_indices(self, text: str) -> List[Tuple[int, str]]:
        """Generate a list of chunks from a given text, along with their character offsets in the original text. Each chunk will be up to the `capacity`.

        See `chunks` for more information.

        Args:
            text (str): Text to split.

        Returns:
            A list of tuples, one for each chunk. The first item will be the character offset relative
            to the original text. The second item is the chunk itself.
            If `trim` was specified in the text splitter, then each chunk will already be
            trimmed as well.
        """
