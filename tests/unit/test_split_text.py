"""Unit-Tests fuer split_text aus web_ui/gradio_summary_ollama.py.

split_text ist eine reine Funktion, liegt aber in einem Modul mit
Seiteneffekten beim Import. Die Fixture ``app`` (conftest.py) importiert es
deshalb mit Fake-Internet.
"""

import pytest


def test_kurzer_text_ein_chunk(app):
    assert app.split_text("Hallo.\nWelt.", max_chars=100) == ["Hallo.\nWelt.\n"]


def test_absaetze_werden_zusammengefasst_bis_zur_grenze(app):
    text = "\n".join(["a" * 40, "b" * 40, "c" * 40])
    chunks = app.split_text(text, max_chars=90)
    assert chunks == ["a" * 40 + "\n" + "b" * 40 + "\n", "c" * 40 + "\n"]


def test_langer_absatz_wird_an_satzgrenzen_geteilt(app):
    absatz = " ".join(f"Satz {i} hat ein paar Woerter." for i in range(30))
    chunks = app.split_text(absatz, max_chars=100)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.rstrip("\n")) <= 100
        assert chunk.rstrip().endswith(".")


def test_inhalt_bleibt_erhalten(app):
    text = "\n".join(f"Absatz {i}. Mit zwei Saetzen." for i in range(100))
    joined = "".join(app.split_text(text, max_chars=200))
    assert joined.split() == text.split()


@pytest.mark.parametrize("text", ["", "\n\n"])
def test_leerer_text(app, text):
    assert app.split_text(text, max_chars=50) == []


def test_standardgroesse_kommt_aus_config(app):
    assert app.CHUNK_CHARS == 12000
    assert len(app.split_text("x. " * 3000)) == 1     # 9000 Zeichen passen in einen Chunk
