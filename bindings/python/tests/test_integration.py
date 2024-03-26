import pytest
from semantic_text_splitter import MarkdownSplitter, TextSplitter
from tokenizers import Tokenizer


def test_chunks():
    splitter = TextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 4) == ["123\n", "123"]


def test_chunks_range():
    splitter = TextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123",
        "\n123",
    ]


def test_chunks_trim():
    splitter = TextSplitter()
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_hugging_face():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 1) == ["123\n", "123"]


def test_hugging_face_range():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123\n", "123"]


def test_hugging_face_trim():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer(tokenizer)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_str():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = TextSplitter.from_huggingface_tokenizer_str(tokenizer.to_str())
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_file():
    splitter = TextSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json"
    )
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken():
    splitter = TextSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n123"
    assert splitter.chunks(text, 2) == ["123\n", "123"]


def test_tiktoken_range():
    splitter = TextSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(2, 3)) == [
        "123\n",
        "123",
    ]


def test_tiktoken_trim():
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo")
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken_model_error():
    with pytest.raises(Exception):
        TextSplitter.from_tiktoken_model("random-model-name")


def test_custom():
    splitter = TextSplitter.from_callback(lambda x: len(x))
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=3) == ["123", "123"]


def test_markdown_chunks():
    splitter = MarkdownSplitter(trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text, 4) == ["123\n", "\n123"]


def test_markdown_chunks_range():
    splitter = MarkdownSplitter(trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123\n",
        "\n123",
    ]


def test_markdown_chunks_trim():
    splitter = MarkdownSplitter()
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_markdown_hugging_face():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text, 1) == ["123\n", "\n123"]


def test_markdown_hugging_face_range():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, trim_chunks=False)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123\n", "\n123"]


def test_markdown_hugging_face_trim():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer)
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_hugging_face_from_str():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = MarkdownSplitter.from_huggingface_tokenizer_str(tokenizer.to_str())
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_hugging_face_from_file():
    splitter = MarkdownSplitter.from_huggingface_tokenizer_file(
        "tests/bert-base-cased.json"
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_tiktoken():
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text, 2) == ["123\n", "\n123"]


def test_markdown_tiktoken_range():
    splitter = MarkdownSplitter.from_tiktoken_model(
        model="gpt-3.5-turbo", trim_chunks=False
    )
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=(2, 3)) == [
        "123\n",
        "\n123",
    ]


def test_markdown_tiktoken_trim():
    splitter = MarkdownSplitter.from_tiktoken_model("gpt-3.5-turbo")
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_markdown_tiktoken_model_error():
    with pytest.raises(Exception):
        MarkdownSplitter.from_tiktoken_model("random-model-name")


def test_markdown_custom():
    splitter = MarkdownSplitter.from_callback(lambda x: len(x))
    text = "123\n\n123"
    assert splitter.chunks(text=text, chunk_capacity=3) == ["123", "123"]
