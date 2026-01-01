import gradio as gr
from transformers import pipeline
from rouge_score import rouge_scorer

# Summarizer
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small"
)

# ROUGE
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

def summarize(text, max_new_tokens, min_new_tokens, num_beams, length_penalty):
    # T5 expects a task prefix
    text = "summarize: " + text.strip()

    out = summarizer(
        text,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(min_new_tokens),
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        do_sample=False,
        truncation=True,  # important for long inputs
    )[0]["summary_text"]

    # Optional: bullets-ish formatting
    out = out.replace(". ", ".\n- ").strip()
    if not out.startswith("- "):
        out = "- " + out
    return out

def compute_rouge(reference, prediction):
    s1 = scorer.score(reference, prediction)["rouge1"]
    sL = scorer.score(reference, prediction)["rougeL"]
    return (s1.precision, s1.recall, s1.fmeasure,
            sL.precision, sL.recall, sL.fmeasure)

def analyze(input_text, reference_summary, max_new_tokens, min_new_tokens, num_beams, length_penalty):
    pred = summarize(input_text, max_new_tokens, min_new_tokens, num_beams, length_penalty)

    # If no reference is provided, don't fake ROUGE
    if not reference_summary or not reference_summary.strip():
        return pred, None, None, None, None, None, None

    r1_p, r1_r, r1_f, rL_p, rL_r, rL_f = compute_rouge(reference_summary, pred)
    return pred, r1_p, r1_r, r1_f, rL_p, rL_r, rL_f

with gr.Blocks() as demo:
    gr.Markdown("# Summarization with ROUGE Evaluation")
    gr.Markdown("Enter a speech, optionally add a reference summary (ground truth), then generate.")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Speech", lines=18, placeholder="Paste speech text here...")
            reference = gr.Textbox(label="Reference Summary (optional, for ROUGE)", lines=6,
                                   placeholder="Paste the gold summary here (if you have one)...")

        with gr.Column():
            output_text = gr.Textbox(label="Generated Summary", lines=18)

    gr.Markdown("### Generation Settings")
    with gr.Row():
        max_new_tokens = gr.Slider(40, 300, value=160, step=10, label="max_new_tokens")
        min_new_tokens = gr.Slider(10, 200, value=80, step=10, label="min_new_tokens")
    with gr.Row():
        num_beams = gr.Slider(1, 8, value=4, step=1, label="num_beams")
        length_penalty = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="length_penalty")

    gr.Markdown("### ROUGE (computed vs Reference Summary)")
    with gr.Row():
        r1_p = gr.Number(label="ROUGE-1 Precision", interactive=False)
        r1_r = gr.Number(label="ROUGE-1 Recall", interactive=False)
        r1_f = gr.Number(label="ROUGE-1 F1", interactive=False)
    with gr.Row():
        rL_p = gr.Number(label="ROUGE-L Precision", interactive=False)
        rL_r = gr.Number(label="ROUGE-L Recall", interactive=False)
        rL_f = gr.Number(label="ROUGE-L F1", interactive=False)

    btn = gr.Button("Generate Summary")
    btn.click(
        fn=analyze,
        inputs=[input_text, reference, max_new_tokens, min_new_tokens, num_beams, length_penalty],
        outputs=[output_text, r1_p, r1_r, r1_f, rL_p, rL_r, rL_f]
    )

demo.launch()
