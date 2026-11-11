"""Gradio-App: Zusammenfassung mit Ollama + Bewertung mit Claude Haiku 5.5 als Judge
inklusive Kostenrechner fuer die verbrauchten Tokens.

Start aus dem Repo-Root:  python web_ui/gradio_summary_ollama_judge.py
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))  # damit summary_func gefunden wird

import anthropic
import dotenv
import gradio as gr

from summary_func import data, reden, list_models, summarize, get_speech, DEFAULT_MODEL

dotenv.load_dotenv()  # liest ANTHROPIC_API_KEY aus der .env

HERE = Path(__file__).parent
JUDGE_SYSTEM_PROMPT = (HERE / "JUDGE_SYSTEM_PROMPT.txt").read_text(encoding="utf-8")
USER_PROMPT = (HERE / "USER_PROMPT.txt").read_text(encoding="utf-8")

judge_cfg = data.get("judge", {})
JUDGE_MODEL = judge_cfg.get("model", "claude-haiku-5-5")
PRICE_INPUT = judge_cfg.get("price_input_per_mtok", 0.10)    # USD pro 1 Mio. Input-Tokens
PRICE_OUTPUT = judge_cfg.get("price_output_per_mtok", 0.50)  # USD pro 1 Mio. Output-Tokens

client = anthropic.Anthropic()


###########################################
#####   Kostenrechner                 #####
###########################################

def token_cost(input_tokens, output_tokens):
    """Kosten in USD fuer einen Aufruf."""
    return input_tokens / 1e6 * PRICE_INPUT + output_tokens / 1e6 * PRICE_OUTPUT


def cost_markdown(input_tokens, output_tokens, total, n_calls, n_reden):
    cost = token_cost(input_tokens, output_tokens)
    avg = total / n_calls if n_calls else 0
    return (
        f"| | Tokens | Kosten |\n|---|---:|---:|\n"
        f"| Input | {input_tokens:,} | ${input_tokens / 1e6 * PRICE_INPUT:.5f} |\n"
        f"| Output | {output_tokens:,} | ${output_tokens / 1e6 * PRICE_OUTPUT:.5f} |\n"
        f"| **Dieser Aufruf** | {input_tokens + output_tokens:,} | **${cost:.5f}** |\n\n"
        f"Session gesamt: **${total:.5f}** bei {n_calls} Bewertung(en)\n\n"
        f"Hochrechnung für {int(n_reden)} Reden: **${avg * n_reden:.4f}**\n\n"
        f"_Preise: ${PRICE_INPUT}/MTok Input, ${PRICE_OUTPUT}/MTok Output_"
    )


###########################################
#####   LLM as a Judge                #####
###########################################

def parse_json(text):
    """Holt das JSON aus der Antwort, auch wenn es in ```json ... ``` steht."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else None


def judge(politician, summary, n_reden, state):
    """Laesst Haiku die Zusammenfassung bewerten und berechnet die Kosten."""
    total, n_calls = state
    rede = reden.get(politician)
    if not rede or not summary or summary.startswith(("⚠️", "⏳")):
        return "⚠️ Erst eine Zusammenfassung erzeugen.", "", state

    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            temperature=0,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_PROMPT.format(
                redner=politician, rede=rede, zusammenfassung=summary)}],
        )
    except anthropic.APIError as e:
        return f"⚠️ Fehler bei der Anfrage an Anthropic: {e}", "", state

    usage = response.usage
    total += token_cost(usage.input_tokens, usage.output_tokens)
    n_calls += 1
    costs = cost_markdown(usage.input_tokens, usage.output_tokens, total, n_calls, n_reden)

    text = response.content[0].text
    try:
        result = parse_json(text)
    except json.JSONDecodeError:
        result = None
    if not result:
        return f"⚠️ Antwort war kein gültiges JSON:\n\n```\n{text}\n```", costs, (total, n_calls)

    def liste(items):
        return "\n".join(f"- {x}" for x in items) if items else "- keine"

    scores = (
        f"| Kriterium | Note |\n|---|:---:|\n"
        f"| Faktentreue | {result.get('faktentreue')} / 5 |\n"
        f"| Abdeckung | {result.get('abdeckung')} / 5 |\n"
        f"| Struktur | {result.get('struktur')} / 5 |\n\n"
        f"**Begründung:** {result.get('begruendung', '')}\n\n"
        f"**Unbelegte Aussagen:**\n{liste(result.get('unbelegte_aussagen', []))}\n\n"
        f"**Fehlende Punkte:**\n{liste(result.get('fehlende_punkte', []))}"
    )
    return scores, costs, (total, n_calls)


###########################################
#####   UI                            #####
###########################################

models = list_models()
default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None)
keys = sorted(reden.keys())

with gr.Blocks() as demo:
    gr.Markdown("# Bundestag Speech Summarization + LLM as a Judge")
    gr.Markdown(f"Zusammenfassung lokal mit Ollama, Bewertung mit `{JUDGE_MODEL}`.")
    cost_state = gr.State((0.0, 0))  # (Gesamtkosten in USD, Anzahl Bewertungen)

    with gr.Row():
        with gr.Column():
            politician = gr.Dropdown(label="Politician", choices=keys, value=(keys[0] if keys else None))
            model = gr.Dropdown(label="Ollama-Modell", choices=models, value=default_model)
            info = gr.Markdown()
            speech_text = gr.Textbox(label="Original Rede", lines=18, interactive=False)

        with gr.Column():
            summarize_button = gr.Button("Generate Summary", variant="primary")
            output_text = gr.Markdown(label="Generated Summary")
            judge_button = gr.Button("Mit Haiku bewerten")
            scores_md = gr.Markdown()
            gr.Markdown("### Kostenrechner")
            n_reden = gr.Number(label="Hochrechnung für Anzahl Reden", value=20, precision=0)
            cost_md = gr.Markdown()

    summarize_button.click(fn=summarize, inputs=[politician, model], outputs=output_text)
    judge_button.click(fn=judge, inputs=[politician, output_text, n_reden, cost_state],
                       outputs=[scores_md, cost_md, cost_state])
    politician.change(fn=get_speech, inputs=politician, outputs=[speech_text, info])
    demo.load(fn=get_speech, inputs=politician, outputs=[speech_text, info])

if __name__ == "__main__":
    demo.launch()
