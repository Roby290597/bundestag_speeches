"""End-to-End-Tests: die Gradio-App laeuft als echter Webserver und wird
ueber HTTP mit gradio_client bedient, so wie es der Browser auch tut.

* test_*_mit_fake_ollama: kompletter Weg durch Gradio (Server, Queue,
  Streaming, Dropdown-Validierung), aber mit Fake-Ollama und Test-XML.
  Laeuft offline und ohne GPU.
* test_live_*: echtes Plenarprotokoll + echtes Ollama. Nur mit
  ``pytest --run-live`` (und laufendem ``ollama serve``).
"""

import importlib
import socket
import sys

import pytest

pytestmark = pytest.mark.e2e
gradio_client = pytest.importorskip("gradio_client")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start(module):
    port = free_port()
    module.demo.queue().launch(server_name="127.0.0.1", server_port=port,
                               prevent_thread_lock=True, quiet=True)
    return gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)


@pytest.fixture
def client(app):
    c = start(app)
    yield c
    app.demo.close()


def test_api_endpunkte_vorhanden(client):
    endpoints = client.view_api(print_info=False, return_format="dict")["named_endpoints"]
    assert "/summarize" in endpoints
    assert "/get_speech" in endpoints


def test_rede_anzeigen_mit_fake_ollama(client):
    text, info = client.predict("Anna Beispiel", api_name="/get_speech")
    assert "Investitionen in Schulen" in text
    assert "Nächste Rede" in text
    assert info.endswith("1 Abschnitt(e)")


def test_zusammenfassung_mit_fake_ollama(client, fake_ollama):
    fake_ollama.reply = lambda p: "**Thema:** Bildung und Kommunen"

    result = client.predict("Anna Beispiel", "qwen3.5:latest", api_name="/summarize")

    assert result == "**Thema:** Bildung und Kommunen"
    assert len(fake_ollama.chat_calls) == 1


def test_streaming_mit_fake_ollama(client, fake_ollama):
    fake_ollama.reply = lambda p: "Ein etwas laengerer Text, der in vielen Token kommt."
    job = client.submit("Anna Beispiel", "qwen3.5:latest", api_name="/summarize")
    job.result(timeout=30)
    zwischenstaende = job.outputs()
    assert len(zwischenstaende) > 1
    assert zwischenstaende[-1] == "Ein etwas laengerer Text, der in vielen Token kommt."


def test_ungueltiger_redner_wird_von_gradio_abgelehnt(client):
    # Das Dropdown erlaubt nur Werte aus `choices`, Gradio meldet einen Fehler
    with pytest.raises(Exception):
        client.predict("Erfundene Person", "qwen3.5:latest", api_name="/summarize")


# --------------------------------------------------------------------------

@pytest.mark.live
def test_live_echtes_protokoll_und_ollama():
    import requests

    try:
        requests.get("http://localhost:11434/api/tags", timeout=3).raise_for_status()
    except requests.RequestException:
        pytest.skip("Ollama laeuft nicht (ollama serve)")

    sys.modules.pop("gradio_summary_ollama", None)
    module = importlib.import_module("gradio_summary_ollama")
    assert len(module.reden) > 10, "Protokoll sollte viele Reden enthalten"
    assert module.default_model, "kein Ollama-Modell installiert"

    client = start(module)
    try:
        redner = sorted(module.reden, key=lambda n: len(module.reden[n]))[0]  # kuerzeste Rede
        result = client.predict(redner, module.default_model, api_name="/summarize")
        assert "**Thema:**" in result
        assert "⚠️" not in result
    finally:
        module.demo.close()


