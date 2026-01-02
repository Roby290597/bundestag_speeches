import gradio as gr
from transformers import pipeline
from rouge_score import rouge_scorer
import yaml



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



def analyze(text):
    summary_text = summarize(text)
    scores = scorer.score(text, summary_text)
    return summary_text, scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure



with gr.Blocks() as demo:
    gr.Markdown("# Summarization with Rouge Evaluation")
    gr.Markdown("Enter speech to obtain bullet points")
    with gr.Row():
        with gr.Column():
            #gr.Dropdown(label="Politician", choices=[], value="Text")
            input_text = gr.Textbox(label="Input Speech", lines=15, placeholder="Enter the speech text here...")
        with gr.Column():
            output_text = gr.Textbox(label="Generated Summary", lines=15)
            with gr.Row():
                with gr.Column():       
                    gr.Markdown("Calculated Rouge Scores")
                    precision = gr.Number(label="Rouge-1 Precision", interactive=True, elem_id="rouge1_precision")
                    recall = gr.Number(label="Rouge-1 Recall", interactive=True, elem_id="rouge1_recall")
                    f1_score = gr.Number(label="Rouge-1 F1", interactive=True, elem_id="rouge1_f1")
    summarize_button = gr.Button("Generate Summary")
    summarize_button.click(fn=analyze, inputs=input_text, outputs=(output_text, precision, recall,f1_score)  )



# Launch the app with Gradio UI
if __name__ == "__main__":
    demo.launch(footer_links=["gradio"])



### Stelle heute (31.12.2025 noc eine vollständige UI fertig    )