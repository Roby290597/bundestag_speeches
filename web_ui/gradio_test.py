import gradio as gr

def greet(name):
    return f"Hello, {name}!"


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Greeting App  
        Enter your name and receive a greeting!
        """
    )
    name_input = gr.Textbox(label="Enter your name")
    greeting_output = gr.Textbox(label="Greeting")
    greet_button = gr.Button("Greet")
    greet_button.click(greet, 
                       inputs=name_input, 
                       outputs=greeting_output,
                       api_name="greet")
                       


demo.launch()


# from gradio_client import Client

# client = Client("http://127.0.0.1:7860/")
# result = client.predict(
# 	name="Roby ",
# 	api_name="/greet"
# )
# print(result)