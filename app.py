import time
import numpy as np
import pandas as pd
from requests import options 
import gradio as gr
    
from utils.utils import download_paper_data, download_titles_and_abstracts, calculate_score
from utils.score import Evaluator, Paper, Score

COOLDOWN_SECONDS = 10   
user_last_request = {} # to track last request time per user

evaluator = Evaluator(online=True)

def wraper_fuc(choice_key, session_id):
    yield from process_id(choice_collection[choice_key], session_id)

def process_id(id_number, session_id):
    """
    Main processing function that:
    1. Fetches data from API
    """
    now = time.time()
    last_time = user_last_request.get(session_id, 0)
    if now - last_time < COOLDOWN_SECONDS:
        raise gr.Error(f"⏱ Please wait {int(COOLDOWN_SECONDS - (now - last_time))}s before submitting again.")
    user_last_request[session_id] = now

    no_df = pd.DataFrame(columns=["Score", "Raw value", "Normalized value"])

    status_message = ""  
    if not id_number:
        status_message += "💀 Please enter an ID number \n"
        yield status_message, "", no_df
        return 
    if not id_number.startswith("W"):
        status_message += "💀 Please enter a valid OpenAlex ID starting with 'W' \n"
        yield status_message, "", no_df
        return 
    
    # Step 1: Fetch data from API
    status_message += "Fetching paper metadata from API... may take a while. \n"
    yield status_message, "", no_df

    paper = Paper(id_number)
    paper = evaluator.fetch_paper_data(paper)
    
    if paper.status != "OK":
        status_message += "☠️ Error fetching paper metadata. Please check the OpenAlex ID and try again.\n"
        yield status_message, "", no_df
        return  

    api_data_display = f"""
  **API Data Retrieved:**

  **Title:** {paper.title if paper.title is not None else 'N/A'}

  **Abstract:** {paper.abstract[:300] + '...' if paper.abstract is not None else 'N/A'}

  **Referenced Works:** {len(paper.references) if paper.references is not None else 'N/A'}
"""

    status_message += "✅ Paper metadata fetched successfully. Now processing...\n"
    yield status_message, api_data_display, no_df

    # Step 2: Process referenced works (fetch titles and abstracts)         
    status_message += "Fetching referenced works data... this may take a while depending on the number of references.\n" 
    yield status_message, api_data_display, no_df 
    
    i = 0
    process_status = ""
    for paper in evaluator.fetch_ref_data(paper):
        i += 1
        if i % 10 == 0:
            process_status = f"⏳ Referenced works processed: {i}/{len(paper.references)} \n"
            yield status_message + process_status, api_data_display, no_df
    status_message += process_status
        
    if paper.status != "OK":
        status_message += f"☠️ {paper.status} Cannot proceed further.\n"
        yield status_message, api_data_display, no_df
        return
    
    status_message += "✅ Referenced works data fetched successfully. Now calculating embeddings...\n"
    yield status_message, api_data_display, no_df

    # Step 3: Calculate embeddings 
    status_message += "Calculating embeddings for the paper and its references... please be patient ⌛\n"
    yield status_message, api_data_display, no_df
    for status in evaluator.calculate_embeddings(paper):
        yield status_message + status, api_data_display, no_df
    status_message += status

    if paper.status != "OK":
        status_message += "☠️ Error calculating embeddings. Cannot proceed further.\n"
        yield status_message, api_data_display, no_df
        return
    
    status_message += "✅ Embeddings calculated successfully. Now evaluating the score...\n"
    yield status_message, api_data_display, no_df


    score = Score()
        
    result = {}
    result["paper_ref"] = score.eval_paper_ref_dissimilarity(paper)
    result["ref_ref"] = score.eval_ref_ref_dissimilarity(paper)
    result["ref_spread"] = score.eval_ref_spread(paper)  

    status_message += "✅ Score evaluation complete!\n"
    yield status_message, api_data_display, no_df

    time.sleep(0.5)

    score_df = pd.DataFrame([
        {"Score": "Score paper-ref", "Raw value": result["paper_ref"], "Normalized value": None},
        {"Score": "Score ref-ref", "Raw value": result["ref_ref"], "Normalized value": None},
        {"Score": "Score ref-std", "Raw value": result["ref_spread"], "Normalized value": None}
    ])


    result_message = f"🎉 Processing complete! Score calculated successfully."
    
    yield status_message + result_message, api_data_display, score_df

with open("texts/intro.md", "r") as f:
    intro_markdown = f.read()

# Create Gradio interface
with gr.Blocks(title="TRUST Score Calculator") as demo:
    
    gr.Markdown(intro_markdown)
    
    session_state = gr.State() 

    choice_collection = {
        "🍎 Attention is all you need": "W2626778328",
        "🍏 NAS-Bench": "W3081305497",
        "🍋 Sensor Data Air Pollution": "W2498521749"
    }

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                id_input = gr.Textbox(
                    label="Enter OpenAlex ID (e.g. W3081305497)", 
                    placeholder="e.g., W3081305497",
                    lines=1,
                    scale=3
                )
                submit_btn = gr.Button("Calculate Score", variant="primary", scale=1)
            with gr.Row():
                gr.Markdown("""
                Or pick an example ID:
                """)
            with gr.Row():
                id_input_alt = gr.Dropdown(
                    choices=list(choice_collection.keys()),
                    label="Pick a fruit", scale=3
                )
                go_btn = gr.Button("Go!", scale=1)  

        with gr.Column(scale=2):

            df = gr.DataFrame(
                headers=["Score", "Raw value", "Normalized value"],
                label="Score Outputs"
            )  


    status_output = gr.Textbox(
        label="Status",
        lines=2
    )
    
    api_data_output = gr.Markdown(
        label="API Data"
    )
    
    # Set up the event handler
    submit_btn.click(
        fn=process_id,
        inputs=[id_input, session_state],
        outputs=[status_output, api_data_output, df]
    )

    go_btn.click(
        fn=wraper_fuc,
        inputs=[id_input_alt, session_state],
        outputs=[status_output, api_data_output, df]
    )

    # Also allow Enter key to submit
    id_input.submit(
        fn=process_id,
        inputs=[id_input, session_state],
        outputs=[status_output, api_data_output, df]
    )

    
    gr.Markdown("""
    ---
    **Note:** This app uses the specter2 model for calculating embeddings from API data.
    The first run may take longer as the model loads.
    """)


if __name__ == "__main__":
    demo.queue(
        max_size=10            # prevents overload
    )
    CSS_STRING = """
    .gradio-container {
        overflow-y: auto !important;
        height: auto !important;
    }
    """
    #demo.launch(css=CSS_STRING)
    demo.launch(css=CSS_STRING)
