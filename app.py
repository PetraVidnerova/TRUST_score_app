import gradio as gr
import requests

from utils import download_paper_data, download_titles_and_abstracts

def load_model():
    """Load the specter2 model lazily"""
    print("Model loaded successfully!")
    

def fetch_data_from_api(id_number):
    """
    Fetch data from a public API based on ID number.
    Using JSONPlaceholder as an example public API.
    """
    # try:
    #     # Using JSONPlaceholder API as an example
    #     url = f"https://jsonplaceholder.typicode.com/posts/{id_number}"
    #     response = requests.get(url, timeout=10)
    #     response.raise_for_status()
    #     data = response.json()
    #     return data
    # except requests.RequestException as e:
    #     return {"error": f"Failed to fetch data: {str(e)}"}
    return {
        "id": id_number, 
        "score": 75.0
    }




def process_id(id_number):
    """
    Main processing function that:
    1. Fetches data from API
    """
    if not id_number:
        return "Please enter an ID number", "", 0.0
    
    if not id_number.startswith("W"):
        return "Please enter a valid OpenAlex ID starting with 'W'", "", 0.0
    
    # Step 1: Fetch data from API
    status = "Fetching paper metadata from API... may take a while."
    data = download_paper_data(id_number)
    
    print("Data fetched:", data)

    if "error" in data:
        return f"Error: {data['error']}", "", 0.0
    

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
        yield result_message, api_data_display, 0.0
        end = True
    else:
        yield "Now we will process referenced works...", api_data_display, None
        end = False 

    if not end:
        title = data["title"]
        abstract = data["abstract"]

        titles_abstracts = []
        i = 0 
        for item in download_titles_and_abstracts(data.get("referenced_works", [])):
            titles_abstracts.append(item)
            i += 1
            yield f"Now we will process referenced works... ({i}/{len(data.get('referenced_works', []))} processed)", api_data_display, None
   
        if len(titles_abstracts) == 0:
            result_message = "⚠️ No valid referenced works found. Score cannot be calculated."
            yield result_message, api_data_display, 0.0
            end = True
        else:
            yield "Referenced works processed successfully. Now calculating the embeddings... be patient", api_data_display, None

    


    result_message = f"✅ Processing complete! Score calculated successfully."
    
    yield result_message, api_data_display, data.get('score', 0.0)


# Create Gradio interface
with gr.Blocks(title="TRUST Score Calculator") as demo:
    gr.Markdown("""
    # TRUST Score Calculator
    
    Under construction: so far it calculates nothing :)
                
    Enter a valid OpenAlex ID.
    """)
    
    with gr.Row():
        with gr.Column():
            id_input = gr.Textbox(
                label="Enter OpenAlex ID (e.g. WW3081305497)", 
                placeholder="e.g., W3081305497",
                lines=1
            )
            submit_btn = gr.Button("Calculate Score", variant="primary")
        
        with gr.Column():
            score_output = gr.Number(
                label="Calculated Score",
                precision=2
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
        inputs=[id_input],
        outputs=[status_output, api_data_output, score_output]
    )
    
    # Also allow Enter key to submit
    id_input.submit(
        fn=process_id,
        inputs=[id_input],
        outputs=[status_output, api_data_output, score_output]
    )
    
    gr.Markdown("""
    ---
    **Note:** This app uses the specter2 model for calculating embeddings from API data.
    The first run may take longer as the model loads.
    """)


if __name__ == "__main__":
    demo.launch()
