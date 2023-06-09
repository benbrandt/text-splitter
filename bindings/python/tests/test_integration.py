from semantic_text_splitter import CharacterTextSplitter


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
