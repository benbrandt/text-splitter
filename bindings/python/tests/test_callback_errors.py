from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import tree_sitter_python

from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter


@pytest.fixture(params=[TextSplitter, MarkdownSplitter, CodeSplitter])
def make_splitter(request):
    def make(callback):
        if request.param is CodeSplitter:
            return CodeSplitter.from_callback(
                tree_sitter_python.language(), callback, 3
            )
        return request.param.from_callback(callback, 3)

    return make


@pytest.fixture(
    params=["chunks", "chunk_indices", "chunk_all", "chunk_all_indices"]
)
def split(request):
    def run(splitter, text="123\n123"):
        if request.param.startswith("chunk_all"):
            return getattr(splitter, request.param)([text, text])
        return getattr(splitter, request.param)(text)

    return run


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_callback_exception(make_splitter, split, error_type):
    error = error_type("callback failed")
    calls = 0
    fail = True

    def callback(text):
        nonlocal calls
        calls += 1
        if fail:
            raise error
        return len(text)

    splitter = make_splitter(callback)
    with pytest.raises(error_type, match="callback failed") as exc:
        split(splitter)
    assert exc.value is error
    assert calls == 1
    assert exc.traceback[-1].name == "callback"

    # Errors belong to one operation, not to the reusable splitter.
    fail = False
    assert split(splitter)


@pytest.mark.parametrize(
    "value,error",
    [
        (None, TypeError),
        ("3", TypeError),
        (1.5, TypeError),
        (-1, OverflowError),
        (2**128, OverflowError),
    ],
)
def test_callback_invalid_size(make_splitter, split, value, error):
    with pytest.raises(error):
        split(make_splitter(lambda _: value))


def test_callback_size_conversion_exception(make_splitter, split):
    error = RuntimeError("size conversion failed")

    class InvalidSize:
        def __index__(self):
            raise error

    with pytest.raises(RuntimeError, match="size conversion failed") as exc:
        split(make_splitter(lambda _: InvalidSize()))
    assert exc.value is error
    assert exc.traceback[-1].name == "__index__"


def test_non_callable_callback(make_splitter, split):
    with pytest.raises(TypeError):
        split(make_splitter(None))


def test_callback_late_exception(make_splitter, split):
    calls = 0

    def callback(text):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("later failure")
        return len(text)

    with pytest.raises(RuntimeError, match="later failure"):
        split(make_splitter(callback), "123\n" * 20)
    assert calls == 3


def test_callback_reentrant_error(make_splitter, split):
    nested = False
    error = ValueError("nested failure")

    def callback(text):
        nonlocal nested
        if nested:
            raise error
        nested = True
        try:
            with pytest.raises(ValueError) as exc:
                split(splitter)
            assert exc.value is error
        finally:
            nested = False
        return len(text)

    splitter = make_splitter(callback)
    assert split(splitter)


def test_callback_thread_isolation(make_splitter):
    barrier = Barrier(2)

    def callback(text):
        barrier.wait(timeout=10)
        raise ValueError(text)

    splitter = make_splitter(callback)

    def run(text):
        with pytest.raises(ValueError) as exc:
            splitter.chunks(text)
        return str(exc.value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, ["aaa", "bbb"])) == ["aaa", "bbb"]


@pytest.mark.parametrize("method", ["chunk_all", "chunk_all_indices"])
def test_callback_batch_stops_at_failing_document(make_splitter, method):
    seen = []
    error = RuntimeError("second document failed")

    def callback(text):
        seen.append(text)
        if "b" in text:
            raise error
        return len(text)

    splitter = make_splitter(callback)
    with pytest.raises(RuntimeError) as exc:
        getattr(splitter, method)(["aaa", "bbb", "ccc"])
    assert exc.value is error
    assert any("a" in text for text in seen)
    assert not any("c" in text for text in seen)
