"""Integrationstests: XML-Download -> Reden-Dict -> Chunking -> Ollama-Streaming.

Hier laufen mehrere Bausteine zusammen (redner_extraction, config.yaml,
split_text, ollama_stream, summarize). Nur die Aussenwelt (Bundestag-Server
und Ollama) ist durch FakeOllama aus conftest.py ersetzt.
"""

import pytest


def run(generator):
    """Sammelt alle Zwischenstaende eines Gradio-Generators."""
    return list(generator)


# ---------------- Laden beim Import ----------------

def test_reden_werden_pro_redner_zusammengefuehrt(app):
    assert sorted(app.reden) == ["Anna Beispiel", "Bernd Muster"]   # Clara Leer hat keinen Text
    anna = app.reden["Anna Beispiel"]
    assert "Investitionen in Schulen" in anna
    assert "\n\nNächste Rede:\nIch möchte noch etwas zur Digitalisierung sagen." in anna


def test_config_wird_gelesen(app):
    assert app.DEFAULT_MODEL == "qwen3.5:latest"
    assert app.NUM_CTX == 8192
    assert app.TEMPERATURE == 0.2


def test_modelle_von_ollama(app):
    assert app.models == ["llama3:latest", "qwen3.5:latest"]
    assert app.default_model == "qwen3.5:latest"


def test_list_models_wenn_ollama_nicht_laeuft(app, fake_ollama):
    fake_ollama.down = True
    assert app.list_models() == []


def test_get_speech(app):
    text, info = app.get_speech("Bernd Muster")
    assert text == "Die Schuldenbremse muss eingehalten werden."
    assert info == "5 Wörter, 1 Abschnitt(e)"


# ---------------- summarize: kurze Rede ----------------

def test_kurze_rede_ein_einziger_llm_aufruf(app, fake_ollama):
    fake_ollama.reply = lambda payload: "**Thema:** Haushalt\n\n**Kernaussagen:**\n- Schuldenbremse"

    outputs = run(app.summarize("Bernd Muster", "qwen3.5:latest"))

    assert len(fake_ollama.chat_calls) == 1
    assert outputs[-1] == "**Thema:** Haushalt\n\n**Kernaussagen:**\n- Schuldenbremse"
    # Streaming: die Ausgabe waechst schrittweise
    assert len(outputs) > 1
    assert all(outputs[i + 1].startswith(outputs[i]) for i in range(len(outputs) - 1))


def test_payload_an_ollama(app, fake_ollama):
    run(app.summarize("Bernd Muster", "llama3:latest"))
    payload = fake_ollama.chat_calls[0]

    assert payload["model"] == "llama3:latest"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert payload["options"] == {"num_ctx": 8192, "temperature": 0.2}
    system, user = payload["messages"]
    assert system == {"role": "system", "content": app.SYSTEM_PROMPT}
    assert "Bernd Muster" in user["content"]
    assert "Die Schuldenbremse muss eingehalten werden." in user["content"]
    assert "**Kernaussagen:**" in user["content"]


# ---------------- summarize: lange Rede (Map-Reduce) ----------------

@pytest.fixture
def lange_rede(app, monkeypatch):
    absatz = " ".join(f"Satz {i} ueber den Haushalt." for i in range(250))   # ca. 7000 Zeichen
    text = "\n".join([absatz] * 4)
    monkeypatch.setitem(app.reden, "Lang Redner", text)
    return text


def test_lange_rede_map_reduce(app, fake_ollama, lange_rede):
    n_chunks = len(app.split_text(lange_rede))
    assert n_chunks > 1
    fake_ollama.reply = lambda p: f"Teil-{len(fake_ollama.chat_calls)}"

    outputs = run(app.summarize("Lang Redner", "qwen3.5:latest"))

    # ein Aufruf pro Abschnitt + ein finaler Aufruf
    assert len(fake_ollama.chat_calls) == n_chunks + 1
    fortschritt = [o for o in outputs if o.startswith("⏳")]
    assert fortschritt == [f"⏳ Fasse Abschnitt {i}/{n_chunks} zusammen ..." for i in range(1, n_chunks + 1)]

    # der finale Prompt enthaelt die Teilzusammenfassungen statt der Originalrede
    final_prompt = fake_ollama.chat_calls[-1]["messages"][1]["content"]
    assert "Zusammenfassungen der einzelnen Abschnitte" in final_prompt
    for i in range(1, n_chunks + 1):
        assert f"Abschnitt {i}:\nTeil-{i}" in final_prompt
    assert "Satz 17 ueber den Haushalt" not in final_prompt
    assert outputs[-1] == f"Teil-{n_chunks + 1}"


# ---------------- summarize: Fehlerfaelle ----------------

def test_fallback_fuer_modelle_ohne_think(app, fake_ollama):
    fake_ollama.reject_think = True
    outputs = run(app.summarize("Bernd Muster", "llama3:latest"))

    assert len(fake_ollama.chat_calls) == 2
    assert "think" in fake_ollama.chat_calls[0]
    assert "think" not in fake_ollama.chat_calls[1]
    assert outputs[-1] == "**Thema:** Bildung"


def test_fehler_im_stream(app, fake_ollama):
    fake_ollama.stream_error = "model not found"
    outputs = run(app.summarize("Bernd Muster", "gibtsnicht"))
    assert outputs[-1] == "⚠️ Fehler bei der Anfrage an Ollama: model not found"


def test_ollama_nicht_erreichbar(app, fake_ollama):
    fake_ollama.down = True
    outputs = run(app.summarize("Bernd Muster", "qwen3.5:latest"))
    assert outputs[-1].startswith("⚠️ Fehler bei der Anfrage an Ollama")


def test_unbekannter_redner(app, fake_ollama):
    assert run(app.summarize("Niemand", "qwen3.5:latest")) == ["⚠️ Keine Rede gefunden."]
    assert fake_ollama.chat_calls == []


def test_kein_modell(app, fake_ollama):
    outputs = run(app.summarize("Bernd Muster", None))
    assert outputs == ["⚠️ Kein Ollama-Modell gefunden. Läuft `ollama serve`?"]
    assert fake_ollama.chat_calls == []
