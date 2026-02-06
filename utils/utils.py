import logging 
import requests
import torch

from tenacity import (
    retry, stop_after_attempt, retry_if_exception_type, 
    before_sleep_log, wait_random_exponential
)

logger = logging.getLogger("__main__")

@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry_error_callback=lambda _: None
)
def send_request(url, params, timeout):
    params["mailto"] = "petra@cs.cas.cz"
    response = requests.get(
        url,
        params=params,
        timeout=timeout
    )
    if response.status_code == 404: 
        logger.warning(f"Data not found at {url}.")
        return None
    response.raise_for_status()
    data = response.json()
    return data

def eat_prefix(alexid):
    PREFIX = "https://openalex.org/"
    if alexid.startswith(PREFIX):
        return alexid[len(PREFIX):]
    else:
        return alexid


def download_titles_and_abstracts(works):
    for work_id in works:
        data = download_paper_data(work_id, select="title,abstract_inverted_index")
        if data is not None and "title" in data and "abstract" in data:
            yield (data["title"], data["abstract"])


def download_paper_data(alexid, select: str):
    base_url = "https://api.openalex.org/works/"
    full_url = base_url + eat_prefix(alexid)
    params = {
        "select": select
    }
    timeout = 10
    data = send_request(full_url, params, timeout)
    if data is None:
        return 
    
    if "abstract_inverted_index" in data:
        if data["abstract_inverted_index"]:
            data["abstract"] = create_abstract(data["abstract_inverted_index"])
            del data["abstract_inverted_index"]
        else:
            data["abstract"] = None
            del data["abstract_inverted_index"]
    return data

def create_abstract(abstract_index):
    if abstract_index is None:
        return None
    maximum = 0
    for indexes in abstract_index.values():
        m = max(indexes)
        if m > maximum:
            maximum = m
    words = [""] * (maximum+1)
    for w, indexes in abstract_index.items():
        for i in indexes:
            words[i] = w
    return " ".join(words)

def calculate_score(paper_embedding, ref_embeddings):
    if ref_embeddings.shape[0] == 0:
        return 0.0
    if paper_embedding is None:
        paper_embedding = ref_embeddings.mean(axis=0).unsqueeze(0)
    
    paper_normed = torch.nn.functional.normalize(paper_embedding, p=2, dim=1)   # (1, n)
    ref_normed = torch.nn.functional.normalize(ref_embeddings, p=2, dim=1)   # (m, n)

    cosine_sim_matrix = ref_normed @ paper_normed.T   # (m, 1)
    score = cosine_sim_matrix.flatten().mean().item()

    # similarities = torch.nn.functional.cosine_similarity(
    #     paper_embedding,
    #     ref_embeddings,
    #     dim=1
    # )
    # score = similarities.mean().item()
    return 1.0 - score
