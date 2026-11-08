import os
os.environ.setdefault("USE_TF", "0")  # transformers soll TensorFlow/Keras 3 nicht laden

import gradio as gr
import yaml
import json
import re
## XML
import xml.etree.ElementTree as ET
import requests
import os, sys
sys.path.insert(0, os.getcwd())  # Aktuelles Verzeichnis zu sys.path hinzufügen

from extr.redner_extraction import extract_speeches



# Load YAML data
with open("config.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

ollama_cfg = data.get("ollama", {})
OLLAMA_HOST = ollama_cfg.get("host", "http://localhost:11434")
DEFAULT_MODEL = ollama_cfg.get("model", "llama3:latest")
NUM_CTX = ollama_cfg.get("num_ctx", 8192)
CHUNK_CHARS = ollama_cfg.get("chunk_chars", 12000)
TEMPERATURE = ollama_cfg.get("temperature", 0.2)

url = data['data']["xml_file"]
response = requests.get(url)
# XML in einen Tree parsen
root = ET.fromstring(response.content)

reden = {}
for speech in extract_speeches(root):
    if not speech['text'].strip():
        continue
    if speech['name'] not in reden:
        reden[speech['name']] = speech['text']
    else:
        reden[speech['name']] += "\n\nNächste Rede:\n" + speech['text']






###########################################
#####   Ollama                        #####
###########################################

SYSTEM_PROMPT = """Du bist ein neutraler politischer Analyst und fasst Reden aus dem Deutschen Bundestag zusammen.
Regeln:
- Antworte ausschließlich auf Deutsch.
- Verwende nur Informationen, die im Text stehen. Erfinde nichts und bewerte nicht.
- Zwischenrufe, Beifall und Begrüßungsformeln ignorierst du."""

CHUNK_PROMPT = """Hier ist Abschnitt {i} von {n} einer Bundestagsrede von {name}.
Fasse die wichtigsten Aussagen, Forderungen und Argumente dieses Abschnitts in knappen Stichpunkten zusammen.

Abschnitt:
\"\"\"
{text}
\"\"\""""

FINAL_PROMPT = """Fasse die folgende Bundestagsrede von {name} zusammen.
Nutze genau diese Struktur in Markdown:

**Thema:** 
- ein Satz

**Kernaussagen:**
- Maximal 3 Stichpunkte

**Forderungen / Positionen:**
- Stichpunkte (oder "keine konkreten Forderungen")

**Kritik an anderen Parteien / der Regierung:**
- Stichpunkte (oder "keine")

{kind}:
\"\"\"
{text}
\"\"\""""


def list_models():
    try:
        tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).json()
        return sorted(m["name"] for m in tags.get("models", []))
    except requests.RequestException:
        return []


def ollama_stream(prompt, model):
    """Streamt die Antwort von Ollama Token für Token."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "think": False,  # Thinking-Modelle (qwen3.x, deepseek-r1) sollen direkt antworten
        "options": {"num_ctx": NUM_CTX, "temperature": TEMPERATURE},
    }
    r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=600)
    if r.status_code == 400 and "think" in r.text:
        # Modell unterstützt den think-Parameter nicht (z.B. llama3)
        payload.pop("think")
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, stream=True, timeout=600)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        yield chunk.get("message", {}).get("content", "")


def split_text(text, max_chars=CHUNK_CHARS):
    """Teilt den Text an Absatz- bzw. Satzgrenzen in Stücke von höchstens max_chars Zeichen."""
    parts = []
    for para in text.split("\n"):
        if len(para) <= max_chars:
            parts.append(para)
        else:
            parts.extend(re.split(r"(?<=[.!?])\s+", para))

    chunks, current = [], ""
    for part in parts:
        if current and len(current) + len(part) + 1 > max_chars:
            chunks.append(current)
            current = ""
        current += part + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def summarize(politician, model):
    """Map-Reduce: lange Reden erst abschnittsweise, dann gesamt zusammenfassen."""
    text = reden.get(politician)
    if not text:
        yield "⚠️ Keine Rede gefunden."
        return
    if not model:
        yield "⚠️ Kein Ollama-Modell gefunden. Läuft `ollama serve`?"
        return

    chunks = split_text(text)
    try:
        if len(chunks) == 1:
            final_input, kind = text, "Rede"
        else:
            partials = []
            for i, chunk in enumerate(chunks, 1):
                yield f"⏳ Fasse Abschnitt {i}/{len(chunks)} zusammen ..."
                prompt = CHUNK_PROMPT.format(i=i, n=len(chunks), name=politician, text=chunk)
                partials.append("".join(ollama_stream(prompt, model)))
            final_input = "\n\n".join(f"Abschnitt {i}:\n{p}" for i, p in enumerate(partials, 1))
            kind = "Zusammenfassungen der einzelnen Abschnitte"

        output = ""
        for token in ollama_stream(FINAL_PROMPT.format(name=politician, kind=kind, text=final_input), model):
            output += token
            yield output
    except (requests.RequestException, RuntimeError) as e:
        yield f"⚠️ Fehler bei der Anfrage an Ollama: {e}"


def get_speech(politician):
    text = reden.get(politician, "⚠️ Keine Rede gefunden.")
    return text, f"{len(text.split())} Wörter, {len(split_text(text))} Abschnitt(e)"

