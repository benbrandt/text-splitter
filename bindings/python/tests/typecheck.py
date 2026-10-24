"""Static API regression tests; run with mypy, not pytest."""

from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter
from typing_extensions import assert_type


class Indexable:
    def __index__(self) -> int:
        return 20


def check_api(language: object) -> None:
    text = TextSplitter((10, 20), overlap=2, trim=False)
    markdown = MarkdownSplitter(20)
    code = CodeSplitter(language, 20)
    assert_type(TextSplitter(Indexable()), TextSplitter)
    assert_type(text.chunk_all(("hello", "world")), list[list[str]])
    for splitter in (text, markdown, code):
        assert_type(splitter.chunks("hello"), list[str])
        assert_type(splitter.chunk_indices("hello"), list[tuple[int, str]])
        assert_type(splitter.chunk_all(["hello"]), list[list[str]])
        assert_type(splitter.chunk_all_indices(["hello"]), list[list[tuple[int, str]]])

    assert_type(TextSplitter.from_callback(len, 20), TextSplitter)
    assert_type(MarkdownSplitter.from_callback(len, 20), MarkdownSplitter)
    assert_type(CodeSplitter.from_callback(language, len, 20), CodeSplitter)
    assert_type(TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 20), TextSplitter)
    assert_type(
        MarkdownSplitter.from_huggingface_tokenizer_str("{}", 20), MarkdownSplitter
    )
    assert_type(
        CodeSplitter.from_huggingface_tokenizer_file(language, "tokenizer.json", 20),
        CodeSplitter,
    )

    # --strict includes unused-ignore checks: these must remain type errors.
    TextSplitter("twenty")  # type: ignore[arg-type]
    MarkdownSplitter.from_callback(123, 20)  # type: ignore[arg-type]
    CodeSplitter.from_callback(language, lambda _: "bad", 20)  # type: ignore[arg-type,return-value]
    text.chunks(123)  # type: ignore[arg-type]
    markdown.chunk_all([123])  # type: ignore[list-item]
    _wrong: list[int] = code.chunks("hello")  # type: ignore[assignment]
