import pytest
from semantic_text_splitter import MarkdownSplitter, TextSplitter
from tokenizers import Tokenizer


def test_chunks() -> None:
    splitter = TextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 4) == ["123\n", "123"]


def test_chunks_range() -> None:
    splitter = TextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123",
        "\n123",
    ]


def test_chunks_trim() -> None:
    splitter = TextSplitter()
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_hugging_face() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 1) == ["123\n", "123"]


def test_hugging_face_range() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123\n", "123"]


def test_hugging_face_trim() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_str() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer_str(tokenizer.to_str())
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_file() -> None:
    splitter = TextSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json"
    )
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken() -> None:
    splitter = TextSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n123"
    assert splitter.chunks(text, 2) == ["123\n", "123"]


def test_tiktoken_range() -> None:
    splitter = TextSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(2, 3)) == [
        "123\n",
        "123",
    ]


def test_tiktoken_trim() -> None:
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo")
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken_model_error() -> None:
    with pytest.raises(Exception):
        TextSplitter.from_tiktoken_model("random-model-name")


def test_custom() -> None:
    splitter = TextSplitter.from_callback(lambda x: len(x))
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=3) == ["123", "123"]


def test_markdown_chunks() -> None:
    splitter = MarkdownSplitter(trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text, 4) == ["123\n", "\n123"]


def test_markdown_chunks_range() -> None:
    splitter = MarkdownSplitter(trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123\n",
        "\n123",
    ]


def test_markdown_chunks_trim() -> None:
    splitter = MarkdownSplitter()
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_markdown_hugging_face() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text, 1) == ["123\n", "\n123"]


def test_markdown_hugging_face_range() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123\n", "\n123"]


def test_markdown_hugging_face_trim() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_hugging_face_from_str() -> None:
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer_str(tokenizer.to_str())
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_hugging_face_from_file() -> None:
    splitter = MarkdownSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json"
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_tiktoken() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text, 2) == ["123\n", "\n123"]


def test_markdown_tiktoken_range() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(2, 3)) == [
        "123\n",
        "\n123",
    ]


def test_markdown_tiktoken_trim() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model("gpt-3.5-turbo")
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_tiktoken_model_error() -> None:
    with pytest.raises(Exception):
        MarkdownSplitter.from_tiktoken_model("random-model-name")


def test_markdown_custom() -> None:
    splitter = MarkdownSplitter.from_callback(lambda x: len(x))
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=3) == ["123", "123"]


def test_char_indices() -> None:
    splitter = TextSplitter()
    text = "123\n123"
    assert splitter.chunk_indices(text=text, chunk_capacity=4) == [
        (0, "123"),
        (4, "123"),
    ]


def test_char_indices_with_multibyte_character() -> None:
    splitter = TextSplitter()
    text = "12ü\n123"
    assert len("12ü\n") == 4
    assert splitter.chunk_indices(text=text, chunk_capacity=4) == [
        (0, "12ü"),
        (4, "123"),
    ]


def test_markdown_char_indices() -> None:
    splitter = MarkdownSplitter()
    text = "123\n456\n789"
    assert splitter.chunk_indices(text=text, chunk_capacity=4) == [
        (0, "123"),
        (4, "456"),
        (8, "789"),
    ]


def test_markdown_char_indices_with_multibyte_character() -> None:
    splitter = MarkdownSplitter()
    text = "12ü\n12ü\n12ü"
    assert len("12ü\n") == 4
    assert splitter.chunk_indices(text=text, chunk_capacity=4) == [
        (0, "12ü"),
        (4, "12ü"),
        (8, "12ü"),
    ]
