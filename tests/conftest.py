"""Gemeinsame Fixtures und Attrappen (Fakes) fuer alle Tests.

Zwei Dinge machen den Code ohne Vorbereitung schwer testbar:

1. ``extr/redner_extraction.py`` laedt beim Import zwei Hugging-Face-Modelle.
   Damit Tests schnell und offline laufen, ersetzen wir ``transformers`` und
   ``sentence_transformers`` in ``sys.modules`` durch kleine Fakes, BEVOR
   irgendein Test die Module importiert. conftest.py wird von pytest immer
   zuerst geladen, deshalb passiert das hier ganz oben.

2. ``web_ui/gradio_summary_ollama.py`` laedt beim Import das Plenarprotokoll
   aus dem Internet und fragt Ollama nach Modellen. Die Fixture ``app``
   ersetzt dafuer ``requests.get`` / ``requests.post`` durch einen Fake-Server.
"""

import copy
import importlib
import json
import os
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
for sub in ("extr", "web_ui"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from sample_data import SITZUNG_XML  # noqa: E402


# --------------------------------------------------------------------------
# Fakes fuer die ML-Bibliotheken
# --------------------------------------------------------------------------

class FakeSentimentPipeline:
    """Gibt fuer jeden Aufruf das naechste Ergebnis aus ``responses`` zurueck."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        if self.responses:
            return [self.responses[(len(self.calls) - 1) % len(self.responses)]]
        return [{"label": "neutral", "score": 0.5}]


class FakeSentenceTransformer:
    """Deterministische Pseudo-Embeddings: gleicher Text -> gleicher Vektor."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=False):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        vecs = np.stack([
            np.random.default_rng(zlib.crc32(t.encode())).normal(size=16) for t in texts
        ])
        return vecs[0] if single else vecs


def _install_fake_ml_modules():
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    fake_tf.AutoModelForSequenceClassification = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: object()
    )
    fake_tf.pipeline = lambda *a, **k: FakeSentimentPipeline()
    sys.modules["transformers"] = fake_tf

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_st


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="Tests mit echtem Internet und echtem Ollama ausfuehren")


def pytest_configure(config):
    # Live-Tests sollen die echten Bibliotheken benutzen, alle anderen die Fakes.
    if not config.getoption("--run-live"):
        _install_fake_ml_modules()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="braucht --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# Fake-HTTP fuer Bundestag-Server und Ollama
# --------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, content=b"", json_data=None, lines=None, text=""):
        self.status_code = status_code
        self.content = content
        self._json = json_data
        self._lines = lines or []
        self.text = text

    def json(self):
        return self._json

    def iter_lines(self):
        yield from self._lines

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")


class FakeOllama:
    """Simuliert die Ollama-HTTP-API (/api/tags und /api/chat mit Streaming).

    ``reply`` ist eine Funktion payload -> Antworttext. Die Antwort wird in
    kleine Token zerlegt und wie bei Ollama als JSON-Zeilen gestreamt.
    Alle empfangenen Payloads landen in ``chat_calls``.
    """

    def __init__(self, models=("qwen3.5:latest", "llama3:latest")):
        self.models = list(models)
        self.chat_calls = []
        self.reply = lambda payload: "**Thema:** Bildung"
        self.reject_think = False   # simuliert Modelle ohne think-Parameter
        self.stream_error = None    # simuliert {"error": ...} im Stream
        self.down = False           # simuliert "ollama serve" laeuft nicht

    def get(self, url, *args, **kwargs):
        if self.down and "/api/" in url:
            raise requests.ConnectionError("Ollama nicht erreichbar")
        if url.endswith("/api/tags"):
            return FakeResponse(json_data={"models": [{"name": m} for m in self.models]})
        # alles andere ist der Download des Plenarprotokolls
        return FakeResponse(content=SITZUNG_XML.encode("utf-8"))

    def post(self, url, json=None, **kwargs):
        if self.down:
            raise requests.ConnectionError("Ollama nicht erreichbar")
        # Kopie, weil ollama_stream das Payload-Dict spaeter veraendert (pop('think'))
        self.chat_calls.append(copy.deepcopy(json))
        if self.reject_think and "think" in json:
            return FakeResponse(status_code=400, text='{"error":"model does not support think"}')
        if self.stream_error:
            lines = [_json_line({"error": self.stream_error})]
        else:
            text = self.reply(json)
            tokens = [text[i:i + 5] for i in range(0, len(text), 5)]
            lines = [_json_line({"message": {"content": t}, "done": False}) for t in tokens]
            lines.append(b"")  # Leerzeilen kommen in echten Streams auch vor
            lines.append(_json_line({"message": {"content": ""}, "done": True}))
        return FakeResponse(lines=lines)


def _json_line(obj):
    return json.dumps(obj).encode("utf-8")


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


@pytest.fixture
def app(fake_ollama):
    """Importiert gradio_summary_ollama frisch, mit Fake-Internet und Fake-Ollama."""
    sys.modules.pop("gradio_summary_ollama", None)
    module = importlib.import_module("gradio_summary_ollama")
    yield module
    sys.modules.pop("gradio_summary_ollama", None)


@pytest.fixture
def sitzung_root():
    import xml.etree.ElementTree as ET
    return ET.fromstring(SITZUNG_XML)
