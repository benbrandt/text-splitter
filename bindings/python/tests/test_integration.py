import pytest
from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter
from tokenizers import Tokenizer  # type: ignore
from tree_sitter import Language
import tree_sitter_python


def test_chunks() -> None:
    splitter = TextSplitter(4, trim=False)
    text = "123\n123"
    assert splitter.chunks(text) == ["123\n", "123"]


def test_chunks_range() -> None:
    splitter = TextSplitter(capacity=(3, 4), trim=False)
    text = "123\n123"
    assert splitter.chunks(text=text) == [
        "123",
        "\n123",
    ]


def test_chunk_overlap() -> None:
    splitter = TextSplitter(capacity=4, overlap=2)
    text = "1234567890"

    assert splitter.chunks(text) == ["1234", "3456", "5678", "7890"]


def test_chunk_overlap_indices() -> None:
    splitter = TextSplitter(capacity=4, overlap=2)
    text = "1234567890"

    assert splitter.chunk_indices(text) == [
        (0, "1234"),
        (2, "3456"),
        (4, "5678"),
        (6, "7890"),
    ]


def test_chunks_trim() -> None:
    splitter = TextSplitter(capacity=4)
    text = "123\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_hugging_face() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, 1, trim=False)
    text = "123\n123"
    assert splitter.chunks(text) == ["123\n", "123"]


def test_hugging_face_range() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(
        tokenizer, capacity=(1, 2), trim=False
    )
    text = "123\n123"
    assert splitter.chunks(text=text) == ["123\n", "123"]


def test_hugging_face_trim() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, 1)
    text = "123\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_hugging_face_from_str() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer_str(tokenizer.to_str(), 1)
    text = "123\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_hugging_face_from_file() -> None:
    splitter = TextSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json", 1
    )
    text = "123\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_tiktoken() -> None:
    splitter = TextSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", capacity=2, trim=False
    )
    text = "123\n123"
    assert splitter.chunks(text) == ["123\n", "123"]


def test_tiktoken_range() -> None:
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", (2, 3), trim=False)
    text = "123\n123"
    assert splitter.chunks(text) == [
        "123\n",
        "123",
    ]


def test_tiktoken_trim() -> None:
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 1)
    text = "123\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_tiktoken_model_error() -> None:
    with pytest.raises(Exception):
        TextSplitter.from_tiktoken_model("random-model-name", 1)


def test_custom() -> None:
    splitter = TextSplitter.from_callback(lambda x: len(x), 3)
    text = "123\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_markdown_chunks() -> None:
    splitter = MarkdownSplitter(4, trim=False)
    text = "123\n\n123"
    assert splitter.chunks(text) == ["123\n", "\n123"]


def test_markdown_chunks_range() -> None:
    splitter = MarkdownSplitter(capacity=(3, 4), trim=False)
    text = "123\n\n123"
    assert splitter.chunks(text=text) == [
        "123\n",
        "\n123",
    ]


def test_markdown_chunks_trim() -> None:
    splitter = MarkdownSplitter(capacity=4)
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_markdown_hugging_face() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, 1, trim=False)
    text = "123\n\n123"
    assert splitter.chunks(text) == ["123\n", "\n123"]


def test_markdown_hugging_face_range() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(
        tokenizer, capacity=(1, 2), trim=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123\n", "\n123"]


def test_markdown_hugging_face_trim() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, capacity=1)
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_markdown_hugging_face_from_str() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer_str(
        tokenizer.to_str(), capacity=1
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_markdown_hugging_face_from_file() -> None:
    splitter = MarkdownSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json", capacity=1
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_markdown_tiktoken() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", capacity=2, trim=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text) == ["123\n", "\n123"]


def test_markdown_tiktoken_range() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", capacity=(2, 3), trim=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text) == [
        "123\n",
        "\n123",
    ]


def test_markdown_tiktoken_trim() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model("gpt-3.5-turbo", 1)
    text = "123\n\n123"
    assert splitter.chunks(text=text) == ["123", "123"]


def test_markdown_tiktoken_model_error() -> None:
    with pytest.raises(Exception):
        MarkdownSplitter.from_tiktoken_model("random-model-name", 1)


def test_markdown_custom() -> None:
    splitter = MarkdownSplitter.from_callback(lambda x: len(x), capacity=3)
    text = "123\n\n123"
    assert splitter.chunks(text) == ["123", "123"]


def test_char_indices() -> None:
    splitter = TextSplitter(4)
    text = "123\n123"
    assert splitter.chunk_indices(text) == [
        (0, "123"),
        (4, "123"),
    ]


def test_char_indices_with_multibyte_character() -> None:
    splitter = TextSplitter(4)
    text = "12ü\n123"
    assert len("12ü\n") == 4
    assert splitter.chunk_indices(text=text) == [
        (0, "12ü"),
        (4, "123"),
    ]


def test_markdown_char_indices() -> None:
    splitter = MarkdownSplitter(capacity=4)
    text = "123\n456\n789"
    assert splitter.chunk_indices(text) == [
        (0, "123"),
        (4, "456"),
        (8, "789"),
    ]


def test_markdown_char_indices_with_multibyte_character() -> None:
    splitter = MarkdownSplitter(4)
    text = "12ü\n12ü\n12ü"
    assert len("12ü\n") == 4
    assert splitter.chunk_indices(text=text) == [
        (0, "12ü"),
        (4, "12ü"),
        (8, "12ü"),
    ]


def test_invalid_chunk_range() -> None:
    with pytest.raises(ValueError):
        splitter = TextSplitter((2, 1))


def test_code_splitter() -> None:
    splitter = CodeSplitter(tree_sitter_python.language(), 40)
    text = """
def foo():
    return 42


def bar():
    return 7
"""
    assert splitter.chunks(text) == [
        "def foo():\n    return 42",
        "def bar():\n    return 7",
    ]


def test_invalid_language() -> None:
    with pytest.raises(ValueError):
        splitter = CodeSplitter(tree_sitter_python.language() + 1, 40)


def test_code_char_indices() -> None:
    splitter = CodeSplitter(tree_sitter_python.language(), capacity=4)
    text = "123\n456\n789"
    assert splitter.chunk_indices(text) == [
        (0, "123"),
        (4, "456"),
        (8, "789"),
    ]


def test_code_char_indices_with_multibyte_character() -> None:
    splitter = CodeSplitter(tree_sitter_python.language(), 4)
    text = "12ü\n12ü\n12ü"
    assert len("12ü\n") == 4
    assert splitter.chunk_indices(text=text) == [
        (0, "12ü"),
        (4, "12ü"),
        (8, "12ü"),
    ]
