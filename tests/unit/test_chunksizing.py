"""Unit-Tests fuer extr/support_func.chunksizing (reine Funktion, keine Abhaengigkeiten)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from extr.support_func import chunksizing


def test_leerer_text_gibt_leere_liste():
    assert chunksizing(100, "") == []
    assert chunksizing(100, "   \n\t ") == []


@pytest.mark.parametrize("size", [0, -5])
def test_ungueltige_chunk_size(size):
    with pytest.raises(ValueError):
        chunksizing(size, "Ein Satz.")


def test_kurzer_text_bleibt_ein_chunk():
    assert chunksizing(100, "Erster Satz. Zweiter Satz!") == ["Erster Satz. Zweiter Satz!"]


def test_whitespace_wird_normalisiert():
    assert chunksizing(100, "  Erster   Satz.\n\nZweiter\tSatz?  ") == ["Erster Satz. Zweiter Satz?"]


def test_saetze_werden_nicht_zerschnitten():
    text = "Das ist Satz eins. Das ist Satz zwei. Das ist Satz drei."
    chunks = chunksizing(40, text)
    assert chunks == ["Das ist Satz eins. Das ist Satz zwei.", "Das ist Satz drei."]
    for chunk in chunks:
        assert chunk.endswith(".")


def test_kein_chunk_ueberschreitet_die_groesse():
    text = " ".join(f"Satz Nummer {i} ist kurz." for i in range(200))
    chunks = chunksizing(120, text)
    assert all(len(c) <= 120 for c in chunks)


def test_kein_text_geht_verloren():
    text = " ".join(f"Satz Nummer {i} ist kurz." for i in range(50))
    assert " ".join(chunksizing(80, text)) == text


def test_ueberlanger_satz_wird_hart_geteilt():
    langer_satz = "a" * 250 + "."
    chunks = chunksizing(100, langer_satz)
    assert [len(c) for c in chunks] == [100, 100, 51]
    assert "".join(chunks) == langer_satz




if __name__ == "__main__":
    test_ueberlanger_satz_wird_hart_geteilt()
