import gradio as gr
from transformers import pipeline
from rouge_score import rouge_scorer
import yaml
## XML 
import xml.etree.ElementTree as ET
import requests
import sys 
import os  

from pathlib import Path

##################################################

extr = next((str(p/"extr") for p in [Path.cwd()] + list(Path.cwd().parents) if (p/"extr").is_dir()), None)
if extr and extr not in sys.path:
    sys.path.insert(0, extr)
print("extr hinzugefügt zu sys.path:", extr)

from redner_extraction import extract_speeches, extract_all_speakers, extract_comments 

# Füge Parent-Dir von `extr` zu sys.path hinzu, damit z.B. config.yaml gefunden werden kann
parent_dir = str(Path(extr).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load YAML data
with open(parent_dir + "\\config.yaml", "r") as file:
    data = yaml.safe_load(file)  # safe_load prevents arbitrary code execution
# Modify data (optional)
url = data['data']["xml_file"]
response = requests.get(url)
# XML in einen Tree parsen
tree = ET.ElementTree(ET.fromstring(response.content))
# Wurzel-Element abrufen
root = tree.getroot()

reden = {}
redner = []
doppelt = []
# Beispiel-Ausgabe
for speech in extract_speeches(root):
    #print(f"Redner: {speech['name']}\nRede: {speech['text'][:200]}...\n")
    if speech['name'] not in redner:
        redner.append(speech['name'])
        reden[speech['name']] = speech['text']
    else:
        reden[speech['name']] += "\n Nächste Rede:" + speech['text']
        doppelt.append(speech['name'])

assert len(redner) == len(reden.keys()), "Fehler: Anzahl Redner ungleich Anzahl Redner-Texte im Dict"

###########################################
###########################################





# using pipeline API for summarization task
summarization = pipeline("summarization", model="t5-small", tokenizer="t5-small", max_length=1100, min_length=700)
# rouge metric
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)



def summarize(text):
    summary = summarization(text)[0]['summary_text']
    summary = summary.replace(".", "\n\n -")  # füge Zeilenumbruch nach jedem Satz hinzu
    return summary

def compute_rouge(input_text, summary_text):
    scores = scorer.score(input_text, summary_text)
    return scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure



def analyze(politician=None, reden_dict=reden):
    if politician is not None and politician in reden_dict:
        text = reden_dict[politician]
    else:
        text = "No speech found for the selected politician."
    summary_text = summarize(text)
    scores = scorer.score(text, summary_text)
    return summary_text, scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure


def get_speech(politician):
    return reden.get(politician, "⚠️ Keine Rede gefunden.")

with gr.Blocks() as demo:
    gr.Markdown("# Speech Summarization with Rouge Evaluation")
    gr.Markdown("Choose a politician and generate bullet points of their speech")

    with gr.Row():
        with gr.Column():
            # IMPORTANT: value muss in choices sein
            keys = list(reden.keys())
            politician = gr.Dropdown(
                label="Politician",
                choices=keys,
                value=(keys[0] if keys else None),
            )

            speech_text = gr.Textbox(
                label="Original Rede",
                lines=18,
                interactive=False,
            )

        with gr.Column():
            output_text = gr.Textbox(label="Generated Summary", lines=15)
            gr.Markdown("Calculated Rouge Scores")
            precision = gr.Number(label="Rouge-1 Precision")
            recall = gr.Number(label="Rouge-1 Recall")
            f1_score = gr.Number(label="Rouge-1 F1")

    # Dropdown -> Rede anzeigen

    summarize_button = gr.Button("Generate Summary")
    summarize_button.click(fn=analyze, inputs=politician, outputs=[output_text, precision, recall, f1_score])
    politician.change(fn=get_speech, inputs=politician, outputs=speech_text)

if __name__ == "__main__":
    demo.launch()



