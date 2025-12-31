import gradio as gr

from transformers import pipeline

# using pipeline API for summarization task
summarization = pipeline("summarization", model="t5-small", tokenizer="t5-small", max_length=1100, min_length=700)


# def summarize(text):
#     inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize(text):
    summary = summarization(text)[0]['summary_text']
    return summary

demo = gr.Interface(
    fn=summarize,
    inputs="text",
    outputs="text",
    title="Summarization of Bundestag Speeches",
    description="Enter text to obtain the bullet points."
)



demo.launch()



### Stelle heute (31.12.2025 noc eine vollständige UI fertig    )