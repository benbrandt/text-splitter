from typing import Any, List, Tuple, Union

class CharacterTextSplitter:
    """Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    ### By Number of Characters

    ```python
    from semantic_text_splitter import CharacterTextSplitter

    # Maximum number of characters in a chunk
    max_characters = 1000
    # Optionally can also have the splitter not trim whitespace for you
    splitter = CharacterTextSplitter(trim_chunks=False)

    chunks = splitter.chunks("your document text", max_characters)
    ```

    ### Using a Range for Chunk Capacity

    You also have the option of specifying your chunk capacity as a range.

    Once a chunk has reached a length that falls within the range it will be returned.

    It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

    ```python
    from semantic_text_splitter import CharacterTextSplitter

    # Optionally can also have the splitter trim whitespace for you
    splitter = CharacterTextSplitter()

    # Maximum number of characters in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
    ```

    Args:
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(self, trim_chunks: bool = True) -> None: ...
    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

        ## Method

        To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:

        1. Split the text by a given level
        2. For each section, does it fit within the chunk size?
        a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
        b. No. Split by the next level and repeat.

        The boundaries used to split the text if using the top-level `split` method, in descending length:

        1. Descending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level.
        2. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
        3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
        4. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
        5. Characters

        Splitting doesn't occur below the character level, otherwise you could get partial
        bytes of a char, which may not be a valid unicode str.

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

class HuggingFaceTextSplitter:
    """Text splitter based on a Hugging Face Tokenizer. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    ### By Number of Tokens

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

    ### Using a Range for Chunk Capacity

    You also have the option of specifying your chunk capacity as a range.

    Once a chunk has reached a length that falls within the range it will be returned.

    It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

    ```python
    from semantic_text_splitter import HuggingFaceTextSplitter
    from tokenizers import Tokenizer

    # Optionally can also have the splitter trim whitespace for you
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer)

    # Maximum number of tokens in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
    ```

    Args:
        tokenizer (Tokenizer): A `tokenizers.Tokenizer` you want to use to count tokens for each
            chunk.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(self, tokenizer, trim_chunks: bool = True) -> None: ...
    @staticmethod
    def from_str(json: str, trim_chunks: bool = True) -> HuggingFaceTextSplitter:
        """Instantiate a new text splitter from the given Hugging Face Tokenizer JSON string.

        Args:
            json (str): A valid JSON string representing a previously serialized
                Hugging Face Tokenizer

        Returns:
            The new text splitter
        """
    @staticmethod
    def from_file(path: str, trim_chunks: bool = True) -> HuggingFaceTextSplitter:
        """Instantiate a new text splitter from the Hugging Face tokenizer file at the given path.

        Args:
            path (str): A path to a local JSON file representing a previously serialized
                Hugging Face tokenizer.

        Returns:
            The new text splitter
        """
    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

        ## Method

        To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:

        1. Split the text by a given level
        2. For each section, does it fit within the chunk size?
        a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
        b. No. Split by the next level and repeat.

        The boundaries used to split the text if using the top-level `split` method, in descending length:

        1. Descending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level.
        2. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
        3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
        4. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
        5. Characters

        Splitting doesn't occur below the character level, otherwise you could get partial
        bytes of a char, which may not be a valid unicode str.

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

class TiktokenTextSplitter:
    """Text splitter based on an OpenAI Tiktoken tokenizer. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    ### By Number of Tokens

    ```python
    from semantic_text_splitter import TiktokenTextSplitter

    # Maximum number of tokens in a chunk
    max_tokens = 1000
    # Optionally can also have the splitter not trim whitespace for you
    splitter = TiktokenTextSplitter("gpt-3.5-turbo", trim_chunks=False)

    chunks = splitter.chunks("your document text", max_tokens)
    ```

    ### Using a Range for Chunk Capacity

    You also have the option of specifying your chunk capacity as a range.

    Once a chunk has reached a length that falls within the range it will be returned.

    It is always possible that a chunk may be returned that is less than the `start` value, as adding the next piece of text may have made it larger than the `end` capacity.

    ```python
    from semantic_text_splitter import TiktokenTextSplitter

    # Optionally can also have the splitter trim whitespace for you
    splitter = TiktokenTextSplitter("gpt-3.5-turbo", trim_chunks=False)

    # Maximum number of tokens in a chunk. Will fill up the
    # chunk until it is somewhere in this range.
    chunks = splitter.chunks("your document text", chunk_capacity=(200,1000))
    ```

    Args:
        model (str): The OpenAI model name you want to retrieve a tokenizer for.
        trim_chunks (bool, optional): Specify whether chunks should have whitespace trimmed from the
            beginning and end or not. If False, joining all chunks will return the original
            string. Defaults to True.
    """

    def __init__(self, model: str, trim_chunks: bool = True) -> None: ...
    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

        ## Method

        To preserve as much semantic meaning within a chunk as possible, a recursive approach is used, starting at larger semantic units and, if that is too large, breaking it up into the next largest unit. Here is an example of the steps used:

        1. Split the text by a given level
        2. For each section, does it fit within the chunk size?
        a. Yes. Merge as many of these neighboring sections into a chunk as possible to maximize chunk length.
        b. No. Split by the next level and repeat.

        The boundaries used to split the text if using the top-level `split` method, in descending length:

        1. Descending sequence length of newlines. (Newline is `\r\n`, `\n`, or `\r`) Each unique length of consecutive newline sequences is treated as its own semantic level.
        2. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
        3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
        4. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
        5. Characters

        Splitting doesn't occur below the character level, otherwise you could get partial
        bytes of a char, which may not be a valid unicode str.

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
