import time
import numpy as np
import pandas as pd
from requests import options 
import gradio as gr
    
from utils.utils import download_paper_data, download_titles_and_abstracts, calculate_score
from utils.embeddings import Embeddings

COOLDOWN_SECONDS = 10   
user_last_request = {} # to track last request time per user

model = Embeddings()


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
    
    if not id_number:
        yield "💀 Please enter an ID number", "", no_df
        return 
    if not id_number.startswith("W"):
        yield "💀 Please enter a valid OpenAlex ID starting with 'W'", "", no_df
        return 
    # Step 1: Fetch data from API
    status = "Fetching paper metadata from API... may take a while."
    yield status, "", no_df
    data = download_paper_data(id_number)
    
    #print("Data fetched:", data)

    if "error" in data:
        yield f"☠️ Error: {data['error']}", "",  no_df
        return

    # Prepare output
    api_data_display = f"""
**API Data Retrieved:**

  **Title:** {data.get('title', 'N/A')}

  **Abstract:** {data.get('abstract', 'N/A')[:300] + '...'}

  **Referenced Works:** {len(data.get('referenced_works', []))}
"""
    
    if "title" not in data:
        result_message = "⚠️ No title found in the API data. Score cannot be calculated."
    elif "abstract" not in data or data["abstract"] is None:
        result_message = "⚠️ No abstract found in the API data. Score cannot be calculated."
    elif len(data.get("referenced_works", [])) == 0:
        result_message = "⚠️ No referenced works found in the API data. Score cannot be calculated."
    else:
        result_message = None
    
    if result_message is not None:
        yield result_message, api_data_display, no_df
        return
    else:
        yield "✅ Now we will process referenced works...", api_data_display, no_df

    title = data["title"]
    abstract = data["abstract"]

    titles_abstracts = []
    i = 0 
    for item in download_titles_and_abstracts(data.get("referenced_works", [])):
        titles_abstracts.append(item)
        i += 1
        if i % 10 == 0:
            yield f"❯❯❯❯ Now we will process referenced works... ({i}/{len(data.get('referenced_works', []))} processed)", api_data_display, no_df

   
    if len(titles_abstracts) == 0:
        result_message = "⚠️ No valid referenced works found. Score cannot be calculated."
        yield result_message, api_data_display,  no_df
        return 
        
    yield "✅ Referenced works processed successfully. Now calculating the embeddings... be patient ⌛", api_data_display, no_df

    for result in model.embed([(title, abstract)], titles_only=False):
        paper_embedding = result
    yield "Calculating embeddings for referenced works... now be really patient ⌛⌛⌛", api_data_display, no_df
    for result in model.embed(titles_abstracts, titles_only=False):
        if isinstance(result, str):
            yield result, api_data_display, no_df
        else:
            ref_embeddings = result

    yield "Calculating the final score...", api_data_display, no_df
    score1 = calculate_score(paper_embedding, ref_embeddings)
    score2 = calculate_score(None, ref_embeddings)
    #score = (score1 * score2) / (score1 + score2)
    score = score1
    normalized_score1 = (score1 - 0.01) / (0.1 - 0.01) # todo adjust min/max based on real data
    normalized_score2 = (score2 - 0.01) / (0.1 - 0.01) # todo adjust min/max based on real data
    combined_score =  np.sqrt(score1 * score2) if score1 > 0 and score2 > 0 else 0
    normalized_combined_score = (combined_score - 0.01) / (0.1 - 0.01) # todo adjust min/max based on real data
    time.sleep(0.5)

    score_df = pd.DataFrame([
        {"Score": "Score paper-ref", "Raw value": score1, "Normalized value": normalized_score1},
        {"Score": "Score ref-ref", "Raw value": score2, "Normalized value": normalized_score2},
        {"Score": "Combined Score", "Raw value": combined_score, "Normalized value": normalized_combined_score}
    ])


    result_message = f"🎉 Processing complete! Score calculated successfully."
    
    yield result_message, api_data_display, score_df

with open("texts/intro.md", "r") as f:
    intro_markdown = f.read()

# Create Gradio interface
with gr.Blocks(title="TRUST Score Calculator") as demo:
    
    gr.Markdown(intro_markdown)
    
    session_state = gr.State() 

    choice_collection = {
        "Attention is all you need": "W2626778328",
        "NAS-Bench": "W3081305497",
        "Sensor Data Air Pollution": "W2498521749"
    }

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                id_input = gr.Textbox(
                    label="Enter OpenAlex ID (e.g. W3081305497)", 
                    placeholder="e.g., W3081305497",
                    lines=1
                )
                submit_btn = gr.Button("Calculate Score", variant="primary")
            with gr.Row():
                gr.Markdown("""
                Or pick an example ID:
                """)
            with gr.Row():
                id_input_alt = gr.Dropdown(
                    choices=list(choice_collection.keys()),
                    label="Pick a fruit"
                )
                go_btn = gr.Button("Go!")  

        with gr.Column(scale=2):

            df = gr.DataFrame(
                headers=["Score", "Raw value", "Normalized value"],
                label="Score Outputs"
            )  
            # )
            # with gr.Row():
            #     with gr.Column():
            #         score_output = gr.Number(
            #             label="Calculated Raw Score",
            #             precision=6
            #         )   

            #     with gr.Column():
            #         normalized_output = gr.Number(
            #             label="Normalized Score (0-1)",
            #             precision=6
            #         )
            # with gr.Row():
            #     with gr.Column():
            #         score2_output = gr.Number(
            #             label="Calculated Raw Score",
            #             precision=6
            #         )   

            #     with gr.Column():
            #         normalized_output2 = gr.Number(
            #             label="Normalized Score (0-1)",
            #             precision=6
            #         )


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
    demo.launch(allowed_paths=["img"])