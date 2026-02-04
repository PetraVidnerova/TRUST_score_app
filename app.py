import gradio as gr
import requests




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
    2. Calculates embeddings using specter2
    3. Computes and returns the score
    """
    if not id_number:
        return "Please enter an ID number", "", 0.0
    
    try:
        # Convert to integer to validate
        id_num = int(id_number)
        if id_num < 1 or id_num > 100:
            return "Please enter an ID between 1 and 100", "", 0.0
    except ValueError:
        return "Please enter a valid integer ID", "", 0.0
    
    # Step 1: Fetch data from API
    status = "Fetching data from API..."
    data = fetch_data_from_api(id_num)
    
    if "error" in data:
        return f"Error: {data['error']}", "", 0.0
    
    # Prepare output
    api_data_display = f"""
**API Data Retrieved:**
- **ID:** {data.get('id', 'N/A')}
- **Score:** {data.get('score', 'N/A')}
"""
    
    result_message = f"✅ Processing complete! Score calculated successfully."
    
    return result_message, api_data_display, data.get('score', 0.0)


# Create Gradio interface
with gr.Blocks(title="TRUST Score Calculator") as demo:
    gr.Markdown("""
    # TRUST Score Calculator
    
    Under construction: so far it calculates nothing :)
                
    Enter a valid OpenAlex ID (without prefix, e.g. W123456789):
    """)
    
    with gr.Row():
        with gr.Column():
            id_input = gr.Textbox(
                label="Enter ID Number (1-100)", 
                placeholder="e.g., 42",
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
