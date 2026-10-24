"""Exercise shared splitters on both GIL-enabled and free-threaded Python."""

import ctypes
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from semantic_text_splitter import CodeSplitter, MarkdownSplitter, TextSplitter


@pytest.fixture(scope="session")
def code_language():
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        import tree_sitter_python

        yield tree_sitter_python.language()
        return

    # Published Python grammar wheels are not yet usable on these runtimes.
    # Load a grammar with no Python dependency and wrap its static pointer in the
    # same capsule accepted from Python grammar packages. Keep the library alive
    # until the session ends.
    filename = {
        "darwin": "libtext_splitter_test_grammar.dylib",
        "win32": "text_splitter_test_grammar.dll",
    }.get(sys.platform, "libtext_splitter_test_grammar.so")
    path = Path(__file__).resolve().parents[3] / "target/test-grammar/debug" / filename
    grammar = ctypes.CDLL(str(path))
    grammar.text_splitter_test_language.restype = ctypes.c_void_p
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule_new.restype = ctypes.py_object
    yield capsule_new(
        grammar.text_splitter_test_language(), b"tree_sitter.Language", None
    )


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires free-threaded Python",
)
def test_import_does_not_enable_gil() -> None:
    # Check in isolation: unrelated extensions imported by pytest must not mask
    # this module's free-threading declaration.
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; assert not sys._is_gil_enabled(); "
                "import semantic_text_splitter; assert not sys._is_gil_enabled()"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "splitter_type", [TextSplitter, MarkdownSplitter, CodeSplitter]
)
@pytest.mark.parametrize("sizer", ["characters", "callback", "huggingface", "tiktoken"])
def test_share_splitter_between_threads(splitter_type, sizer, request) -> None:
    args = (
        (request.getfixturevalue("code_language"),)
        if splitter_type is CodeSplitter
        else ()
    )
    if sizer == "callback":
        splitter = splitter_type.from_callback(*args, len, 12)
    elif sizer == "huggingface":
        path = str(Path(__file__).with_name("bert-base-cased.json"))
        splitter = splitter_type.from_huggingface_tokenizer_file(*args, path, 12)
    elif sizer == "tiktoken":
        splitter = splitter_type.from_tiktoken_model(*args, "gpt-3.5-turbo", 12)
    else:
        splitter = splitter_type(*args, 12)

    texts = ["# Héllo 🌍\n\nSome more text.\n", 'fn café() { println!("🦀"); }\n']
    chunks = [splitter.chunks(text) for text in texts]
    indices = [splitter.chunk_indices(text) for text in texts]
    barrier = Barrier(4)
    free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))

    def split_repeatedly(_):
        barrier.wait(timeout=30)
        if free_threaded:
            assert not sys._is_gil_enabled()
        for _ in range(10):
            assert [splitter.chunks(text) for text in texts] == chunks
            assert [splitter.chunk_indices(text) for text in texts] == indices
            assert splitter.chunk_all(texts) == chunks
            assert splitter.chunk_all_indices(texts) == indices

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(split_repeatedly, range(4)))
