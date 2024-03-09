from typing import Any, Callable, List, Tuple, Union


class TextSplitter:
    """Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

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
    """

    def __init__(self, trim_chunks: bool = True) -> None: ...

    @staticmethod
    def from_huggingface_tokenizer(tokenizer, trim_chunks: bool = True) -> TextSplitter:
        """Instantiate a new text splitter from a Hugging Face Tokenizer instance.

        Args:
            tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
                chunk.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                    beginning and end or not. If False, joining all chunks will return the original
                    string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_str(
        json: str, trim_chunks: bool = True
    ) -> TextSplitter:
        """Instantiate a new text splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_file(
        path: str, trim_chunks: bool = True
    ) -> TextSplitter:
        """Instantiate a new text splitter from the Hugging Face tokenizer file at the given path.

        Args:
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.


        Returns:
            The new text splitter
        """

    @staticmethod
    def from_tiktoken_model(model: str, trim_chunks: bool = True) -> TextSplitter:
        """Instantiate a new text splitter based on an OpenAI Tiktoken tokenizer.

        Args:
            model (str): The OpenAI model name you want to retrieve a tokenizer for.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    @staticmethod
    def from_callback(
        callback: Callable[[str], int], trim_chunks: bool = True
    ) -> TextSplitter:
        """Instantiate a new text splitter based on a custom callback.

        Args:
            callback (Callable[[str], int]): A lambda or other function that can be called. It will be
                provided a piece of text, and it should return an integer value for the size.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new text splitter
        """

    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.


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
            chunk_capacity (int | (int, int)): The capacity of characters in each chunk. If a
                single int, then chunks will be filled up as much as possible, without going over
                that number. If a tuple of two integers is provided, a chunk will be considered
                "full" once it is within the two numbers (inclusive range). So it will only fill
                up the chunk until the lower range is met.

        Returns:
            A list of strings, one for each chunk. If `trim_chunks` was specified in the text
            splitter, then each chunk will already be trimmed as well.
        """


class MarkdownSplitter:
    """Markdown splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

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
    """

    def __init__(self, trim_chunks: bool = True) -> None: ...

    @staticmethod
    def from_huggingface_tokenizer(
        tokenizer, trim_chunks: bool = True
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from a Hugging Face Tokenizer instance.

        Args:
            tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
                chunk.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                    beginning and end or not. If False, joining all chunks will return the original
                    string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_str(
        json: str, trim_chunks: bool = True
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_huggingface_tokenizer_file(
        path: str, trim_chunks: bool = True
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter from the Hugging Face tokenizer file at the given path.

        Args:
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.


        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_tiktoken_model(model: str, trim_chunks: bool = True) -> MarkdownSplitter:
        """Instantiate a new markdown splitter based on an OpenAI Tiktoken tokenizer.

        Args:
            model (str): The OpenAI model name you want to retrieve a tokenizer for.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    @staticmethod
    def from_callback(
        callback: Callable[[str], int], trim_chunks: bool = True
    ) -> MarkdownSplitter:
        """Instantiate a new markdown splitter based on a custom callback.

        Args:
            callback (Callable[[str], int]): A lambda or other function that can be called. It will be
                provided a piece of text, and it should return an integer value for the size.
            trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
                beginning and end or not. If False, joining all chunks will return the original
                string. Defaults to True.

        Returns:
            The new markdown splitter
        """

    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

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
        6. Text nodes within elements
        7. Inline elements such as: emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
        8. Block elements suce as: paragraphs, code blocks, and footnote definitions.
        9. Container blocks such as: table rows, block quotes, list items, and HTML blocks.
        10. Meta containers such as: lists and tables.
        11. Thematic breaks or horizontal rules.
        12. Headings by level
        13. Metadata at the beginning of the document

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
        """
