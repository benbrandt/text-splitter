"""Verify the installed wheel's ABI, shipped stubs, and free-threading behavior."""

import argparse
import importlib
import os
import sys
import sysconfig
from importlib.metadata import distribution
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tag", choices=["cp310-abi3", "cp314-cp314t", "cp315-abi3", "cp315-abi3t"]
    )
    args = parser.parse_args()
    package = distribution("semantic-text-splitter")
    assert f"Tag: {args.tag}-" in package.read_text("WHEEL"), package.read_text("WHEEL")
    stub = Path(__file__).resolve().parents[1] / "semantic_text_splitter.pyi"
    installed_stub = package.locate_file("semantic_text_splitter/__init__.pyi")
    assert installed_stub.read_text(encoding="utf-8") == stub.read_text(
        encoding="utf-8"
    )

    free_threaded = args.tag.endswith("t")
    assert bool(sysconfig.get_config_var("Py_GIL_DISABLED")) == free_threaded
    if free_threaded:
        assert os.environ.get("PYTHON_GIL") != "0", (
            "do not force the GIL off for this check"
        )
        assert not sys._is_gil_enabled()
    importlib.import_module("semantic_text_splitter")
    if free_threaded:
        assert not sys._is_gil_enabled(), "importing the extension enabled the GIL"


if __name__ == "__main__":
    main()
