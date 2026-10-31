"""Unit-Tests fuer extr/redner_extraction.py.

Die ML-Modelle sind in conftest.py durch Fakes ersetzt, getestet wird nur
die eigene Logik: XML-Parsing, Kosinus-Aehnlichkeit, Sentiment-Mittelung.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import redner_extraction as rx
from conftest import FakeSentimentPipeline


# ---------------- extract_speeches ----------------

def test_extract_speeches_findet_alle_reden(sitzung_root):
    speeches = rx.extract_speeches(sitzung_root)
    assert [s["name"] for s in speeches] == [
        "Anna Beispiel", "Bernd Muster", "Anna Beispiel", "Clara Leer",
    ]


def test_extract_speeches_ohne_rednerzeile_und_kommentare(sitzung_root):
    erste = rx.extract_speeches(sitzung_root)[0]
    assert erste["text"] == (
        "Sehr geehrte Frau Präsidentin! Wir brauchen mehr Investitionen in Schulen.\n"
        "Die Kommunen dürfen nicht allein gelassen werden."
    )
    assert "Beifall" not in erste["text"]
    assert "(SPD):" not in erste["text"]


def test_extract_speeches_rede_ohne_inhalt_hat_leeren_text(sitzung_root):
    letzte = rx.extract_speeches(sitzung_root)[-1]
    assert letzte == {"name": "Clara Leer", "text": ""}


def test_extract_speeches_fallback_ohne_name_tag():
    root = ET.fromstring(
        '<x><rede><p klasse="redner">Max Mustermann (FDP): bla</p><p>Text.</p></rede></x>'
    )
    assert rx.extract_speeches(root) == [{"name": "Max Mustermann (FDP)", "text": "Text."}]


def test_extract_speeches_ohne_rednerzeile():
    root = ET.fromstring("<x><rede><p>Nur Text.</p></rede></x>")
    assert rx.extract_speeches(root) == [{"name": "", "text": "Nur Text."}]


# ---------------- extract_all_speakers ----------------

def test_extract_all_speakers(sitzung_root):
    speakers = rx.extract_all_speakers(sitzung_root)
    assert len(speakers) == 3
    assert speakers[0] == {"vorname": "Anna", "nachname": "Beispiel", "fraktion": "SPD", "rolle_kurz": ""}
    assert speakers[2]["rolle_kurz"] == "Bundesministerin"
    assert speakers[2]["fraktion"] == ""


# ---------------- extract_comments ----------------

def test_extract_comments_ordnet_kommentare_den_rednern_zu(sitzung_root):
    comments = rx.extract_comments(sitzung_root)
    assert comments["Anna Beispiel"] == ["(Beifall bei der SPD)"]
    assert comments["Bernd Muster"] == ["(Zuruf von der SPD: Unsinn!)"]
    assert comments["Clara Leer"] == ["(Heiterkeit)"]


@pytest.mark.xfail(raises=NameError, strict=True,
                   reason="Bekannter Bug: Fallback greift auf globale Variable `reden` zu, "
                          "die in redner_extraction.py nicht existiert")
def test_extract_comments_rede_ohne_redner():
    root = ET.fromstring("<x><rede><p>Text.</p><kommentar>(Beifall)</kommentar></rede></x>")
    assert rx.extract_comments(root) == {"Unknown": ["(Beifall)"]}


@pytest.mark.parametrize("tag, expected", [
    ("{http://ns}rede", "rede"),
    ("rede", "rede"),
])
def test_local_name_entfernt_namespace(tag, expected):
    assert rx.local_name(tag) == expected


# ---------------- cosine_sim / embed ----------------

def test_cosine_sim_grenzfaelle():
    a = np.array([1.0, 2.0, 3.0])
    assert rx.cosine_sim(a, a) == pytest.approx(1.0)
    assert rx.cosine_sim(a, 5 * a) == pytest.approx(1.0)       # unabhaengig von der Laenge
    assert rx.cosine_sim(a, -a) == pytest.approx(-1.0)
    assert rx.cosine_sim(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(0.0)


def test_embed_gleicher_text_gleicher_vektor():
    v1, v2, v3 = rx.embed(["Hallo Welt", "Hallo Welt", "Etwas anderes"])
    assert rx.cosine_sim(v1, v2) == pytest.approx(1.0)
    assert rx.cosine_sim(v1, v3) < 0.99


# ---------------- get_average_sentiment ----------------

def test_sentiment_mittelt_ueber_chunks(monkeypatch):
    fake = FakeSentimentPipeline([
        {"label": "negative", "score": 0.9},
        {"label": "negative", "score": 0.7},
        {"label": "positive", "score": 0.5},
    ])
    monkeypatch.setattr(rx, "sentiment_pipeline", fake)

    result = rx.get_average_sentiment("x" * 25, chunk_size=10)   # -> 3 Chunks

    assert fake.calls == ["x" * 10, "x" * 10, "x" * 5]
    assert result["chunks_analyzed"] == 3
    assert result["score"] == pytest.approx(0.7)
    assert result["label"] == "negative"


def test_sentiment_ueberspringt_leere_chunks(monkeypatch):
    fake = FakeSentimentPipeline()
    monkeypatch.setattr(rx, "sentiment_pipeline", fake)
    result = rx.get_average_sentiment("abc" + " " * 10, chunk_size=3)
    assert result["chunks_analyzed"] == 1


def test_sentiment_leerer_text(monkeypatch):
    monkeypatch.setattr(rx, "sentiment_pipeline", FakeSentimentPipeline())
    assert rx.get_average_sentiment("") == {"label": "NEUTRAL", "score": 0, "chunks_analyzed": 0}
