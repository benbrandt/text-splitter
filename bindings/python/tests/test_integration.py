from semantic_text_splitter import (
    CharacterTextSplitter,
    HuggingFaceTokenizerTextSplitter,
)
from tokenizers import Tokenizer


def test_chunks():
    splitter = CharacterTextSplitter()
    text = "123\n123"
    assert splitter.chunks(text, 4) == ["123\n", "123"]


def test_chunks_range():
    splitter = CharacterTextSplitter()
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(3, 4)) == [
        "123",
        "\n123",
    ]


def test_chunks_trim():
    splitter = CharacterTextSplitter(trim_chunks=True)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=4) == ["123", "123"]


def test_hugging_face():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTokenizerTextSplitter(tokenizer)
    text = "123\n123"
    assert splitter.chunks(text, 1) == ["123\n", "123"]


def test_hugging_face_range():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTokenizerTextSplitter(tokenizer)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=(1, 2)) == ["123\n", "123"]


def test_hugging_face_trim():
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    splitter = HuggingFaceTokenizerTextSplitter(tokenizer, trim_chunks=True)
    text = "123\n123"
    assert splitter.chunks(text=text, chunk_capacity=1) == ["123", "123"]
