import pytest
from semantic_text_splitter import (
    CharacterTextSplitter,
    HuggingFaceTextSplitter,
    TiktokenTextSplitter,
)
from tokenizers import Tokenizer


def test_chunks():
    splitter = CharacterTextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 4) == ["123\n", "123"]


def test_chunks_range():
    splitter = CharacterTextSplitter(trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123",
        "\n123",
    ]


def test_chunks_trim():
    splitter = CharacterTextSplitter()
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_hugging_face():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 1) == ["123", "\n123"]


def test_hugging_face_range():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer, trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123", "\n123"]


def test_hugging_face_trim():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter(tokenizer)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_str():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTextSplitter.from_str(tokenizer.to_str())
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_hugging_face_from_file():
    splitter = HuggingFaceTextSplitter.from_file("tests/bert-base-cased.json")
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken():
    splitter = TiktokenTextSplitter(model="gpt-3.5-turbo", trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text, 2) == ["123\n", "123"]


def test_tiktoken_range():
    splitter = TiktokenTextSplitter(model="gpt-3.5-turbo", trim_chunks=False)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(2, 3)) == [
        "123\n",
        "123",
    ]


def test_tiktoken_trim():
    splitter = TiktokenTextSplitter("gpt-3.5-turbo")
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]


def test_tiktoken_model_error():
    with pytest.raises(Exception):
        TiktokenTextSplitter("random-model-name")
