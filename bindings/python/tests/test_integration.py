import asyncio
import time
from contextlib import suppress

import pytest
import tree_sitter_python
from tokenizers import Tokenizer  # type: ignore

from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter


def make_markdown(target_bytes: int) -> str:
    para = "lorem ipsum dolor sit amet consectetur adipiscing elit\n\n"
    blocks: list[str] = []
    total = 0
    i = 0
    while total < target_bytes:
        block = f"## Section {i}\n\n" + para * 3
        blocks.append(block)
        total += len(block.encode())
        i += 1
    return "".join(blocks)


async def record_gaps(gaps: list[float]) -> None:
    last = time.perf_counter()
    while True:
        await asyncio.sleep(0.02)
        now = time.perf_counter()
        gaps.append(now - last)
        last = now


async def max_event_loop_gap_during_worker_call(call) -> tuple[float, float]:
    gaps: list[float] = []
    task = asyncio.create_task(record_gaps(gaps))
    await asyncio.sleep(0.1)

    started_at = time.perf_counter()
    await asyncio.to_thread(call)
    duration = time.perf_counter() - started_at

    await asyncio.sleep(0.1)
    _ = task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    return duration, max(gaps)


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


def test_markdown_tiktoken_chunking_releases_gil_in_worker_thread() -> None:
    splitter = MarkdownSplitter.from_tiktoken_model(
        "gpt-3.5-turbo", capacity=512, overlap=64
    )
    text = make_markdown(2_000_000)

    duration, max_gap = asyncio.run(
        max_event_loop_gap_during_worker_call(lambda: splitter.chunk_indices(text))
    )

    if duration < 0.2:
        pytest.skip("chunking completed too quickly for a reliable GIL timing check")

    assert max_gap < duration * 0.5


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
        _ = TextSplitter((2, 1))


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


def test_invalid_language_type() -> None:
    with pytest.raises(TypeError):
        _ = CodeSplitter(tree_sitter_python.language, 40)  # type: ignore


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


def test_chunk_all() -> None:
    splitter = TextSplitter(4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all(texts)
    assert chunks == [["123", "123"], ["456", "456"]]


def test_chunk_all_indices() -> None:
    splitter = TextSplitter(4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all_indices(texts)
    assert chunks == [[(0, "123"), (4, "123")], [(0, "456"), (4, "456")]]


def test_chunk_all_markdown() -> None:
    splitter = MarkdownSplitter(4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all(texts)
    assert chunks == [["123", "123"], ["456", "456"]]


def test_chunk_all_indices_markdown() -> None:
    splitter = MarkdownSplitter(4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all_indices(texts)
    assert chunks == [[(0, "123"), (4, "123")], [(0, "456"), (4, "456")]]


def test_chunk_all_code() -> None:
    splitter = CodeSplitter(tree_sitter_python.language(), 4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all(texts)
    assert chunks == [["123", "123"], ["456", "456"]]


def test_chunk_all_indices_code() -> None:
    splitter = CodeSplitter(tree_sitter_python.language(), 4)
    texts = ["123\n123", "456\n456"]
    chunks = splitter.chunk_all_indices(texts)
    assert chunks == [[(0, "123"), (4, "123")], [(0, "456"), (4, "456")]]


@pytest.mark.parametrize(
    "splitter_factory",
    [
        lambda: TextSplitter.from_callback(lambda x: len(x), 4),
        lambda: MarkdownSplitter.from_callback(lambda x: len(x), 4),
        lambda: CodeSplitter.from_callback(
            tree_sitter_python.language(), lambda x: len(x), 4
        ),
    ],
)
def test_chunk_all_with_callback(splitter_factory) -> None:
    splitter = splitter_factory()
    texts = ["123\n123", "456\n456"]

    assert splitter.chunk_all(texts) == [["123", "123"], ["456", "456"]]
    assert splitter.chunk_all_indices(texts) == [
        [(0, "123"), (4, "123")],
        [(0, "456"), (4, "456")],
    ]
