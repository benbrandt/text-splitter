from typing import List, Tuple, Union

class CharacterTextSplitter:
    """
    Plain-text splitter. Recursively splits chunks into the largest semantic units that fit within the chunk size. Also will attempt to merge neighboring chunks if they can fit within the given chunk size.

    ### By Number of Characters

    ```python
    from semantic_text_splitter import CharacterTextSplitter

    # Maximum number of characters in a chunk
    max_characters = 1000
    # Optionally can also have the splitter trim whitespace for you
    splitter = CharacterTextSplitter(trim_chunks=True)

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
    ```"""

    def __init__(self, trim_chunks: bool = False) -> None: ...
    def chunks(
        self, text: str, chunk_capacity: Union[int, Tuple[int, int]]
    ) -> List[str]:
        """
        Generate a list of chunks from a given text. Each chunk will be up to the `chunk_capacity`.

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
        """
