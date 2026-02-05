import logging 
import pickle
from pathlib import Path

from torch.nn.functional import cosine_similarity as cosine_similarity
from utils.embeddings import Embeddings
from utils.utils import download_paper_data, send_request, eat_prefix, create_abstract

logger = logging.getLogger("__main__")

class Paper():
    def __init__(self, openalexid:str, title:str=None, abstract:str=None):
        
        self.openalexid = openalexid
        self.title = title
        self.abstract = abstract

        self.references = [] 
        self.ref_data = []  

        self.embedding = None
        self.ref_embeddings = None

        self.status = "OK"
        self.titles_only = False


class Score():

    def __init__(self):
        self.mean_ref = None

    def eval_paper_ref_dissimilarity(self, paper: Paper):
        if self.mean_ref is None:
            self.mean_ref = paper.ref_embeddings.mean(dim=0, keepdim=True)
        similarity = cosine_similarity(paper.embedding, self.mean_ref)
        return  1.0 - similarity.item()  # dis

    def eval_ref_ref_dissimilarity(self, paper: Paper):
        if self.mean_ref is None:
            self.mean_ref = paper.ref_embeddings.mean(dim=0, keepdim=True)
        
        # compute mean similarity of references to the mean reference embedding
        mean_sim = cosine_similarity(paper.ref_embeddings, self.mean_ref).mean()

        return 1.0 - mean_sim.item()  # dissimilarity

    def eval_ref_spread(self, paper: Paper):
        if self.mean_ref is None:
            self.mean_ref = paper.ref_embeddings.mean(dim=0, keepdim=True)
        
        # compute mean similarity of references to the mean reference embedding
        sim = cosine_similarity(paper.ref_embeddings, self.mean_ref)
        dist = 1.0 - sim
        return dist.std().item()  # spread
    
class Evaluator():

    def __init__(self, online:bool=False, api_key:str=None,
                 force_cpu:bool=False):
        """ Load saved data if their exists and we are not online. """
        self.online = online
        self.api_key = api_key
        self.embeddings_model = Embeddings(
            device="cpu" if force_cpu else "auto")

        self.titles_cache = {} # openalexid -> title
        self.abstracts_cache = {} # openalexid -> abstract
        self.ref_data_cache = {} # openalexid -> (list of references)

        self.paper_embeddings_cache = {} # openalexid -> embedding
        self.ref_embeddings_cache = {} # openalexid -> list of reference embeddings

        self.load_cache()

    def load_cache(self):
        """ Load the cache from disk if it exists. """
        if Path("titles_cache.pickle").exists():
            with open("titles_cache.pickle", "rb") as f:
                self.titles_cache = pickle.load(f)
                logger.info("Titles cache loaded successfully.")
        if Path("abstracts_cache.pickle").exists():
            with open("abstracts_cache.pickle", "rb") as f:
                self.abstracts_cache = pickle.load(f)
                logger.info("Abstracts cache loaded successfully.")
        if Path("ref_data_cache.pickle").exists():
            with open("ref_data_cache.pickle", "rb") as f:
                self.ref_data_cache = pickle.load(f)
                logger.info("Reference data cache loaded successfully.")
        if Path("paper_embeddings_cache.pickle").exists():
            with open("paper_embeddings_cache.pickle", "rb") as f:
                self.paper_embeddings_cache = pickle.load(f)
                logger.info("Paper embeddings cache loaded successfully.")
        if Path("ref_embeddings_cache.pickle").exists():
            with open("ref_embeddings_cache.pickle", "rb") as f:
                self.ref_embeddings_cache = pickle.load(f)
                logger.info("Reference embeddings cache loaded successfully.")

    def save_cache(self):
        """ Save the cache to disk. """
        with open("titles_cache.pickle", "wb") as f:
            pickle.dump(self.titles_cache, f)
        with open("abstracts_cache.pickle", "wb") as f:
            pickle.dump(self.abstracts_cache, f)
        with open("ref_data_cache.pickle", "wb") as f:
            pickle.dump(self.ref_data_cache, f)
        
        with open("paper_embeddings_cache.pickle", "wb") as f:
            pickle.dump(self.paper_embeddings_cache, f)
        with open("ref_embeddings_cache.pickle", "wb") as f:
            pickle.dump(self.ref_embeddings_cache, f)
        logger.debug("Cache saved successfully.")

    def fetch_paper_data(self, paper:Paper):
        """ Fetch paper data from OpenAlex API.
        This method checks the cache first before making an API call. It updates the paper object with the fetched data.
        """
        select_fields = []
        if paper.title is None:
            if paper.openalexid in self.titles_cache:
                paper.title = self.titles_cache[paper.openalexid]
                assert paper.title is not None
            else:
                select_fields.append("title")
        if paper.abstract is None:
            if paper.openalexid in self.abstracts_cache:
                paper.abstract = self.abstracts_cache[paper.openalexid]
                assert isinstance(paper.abstract, str)
            else:
                select_fields.append("abstract_inverted_index")
        if paper.openalexid in self.ref_data_cache:
            paper.references = self.ref_data_cache[paper.openalexid]
        else:
            select_fields.append("referenced_works")
        
        if select_fields:
            select_str = ",".join(select_fields)
            data = download_paper_data(paper.openalexid, select=select_str)   
            if data is None:
                paper.status = "Error during fetching data for given OpenAlex ID."
                return paper

            if  paper.title is None:
                paper.title = data.get("title", None)
                if paper.title is None:
                    paper.status = "Title not found."
        
            if paper.abstract is None:
                paper.abstract = data.get("abstract", None)
                if paper.abstract is None or not isinstance(abstract, str):
                    paper.titles_only = True
                    logger.warning(f"Abstract not found for paper {paper.openalexid}. Will calculate score based on titles only.")
    
            paper.references = data.get("referenced_works", [])
            if len(paper.references) == 0:
                paper.status = "No references found."
            self.ref_data_cache[paper.openalexid] = paper.references

        return paper 
    
    def fetch_ref_data_batched(self, paper:Paper):
        if not self.api_key:
            return self.fetch_ref_data(paper)
        
        batch_size = 10
        to_process = [] # papes that has to be fetched
        # first use cache
        for ref in paper.references:
            if ref in self.titles_cache and ref in self.abstracts_cache:
                title = self.titles_cache[ref]
                abstract = self.abstracts_cache[ref]
                paper.ref_data.append((title, abstract))
            else:
                to_process.append(ref)
        # now process in batches
        for i in range(0, len(to_process), batch_size):
            works = to_process[i:i+batch_size]
            works = [eat_prefix(w) for w in works]
           
            url = "https://api.openalex.org/works"
            params = {
                "api_key": self.api_key,
                "filter": "openalex:" + "|".join(works),
                "select": "id,title,abstract_inverted_index"
            }
            data = send_request(url, params, 10)
            if data is None:
                raise ValueError("Error during batched fetching of reference data.")
            for item in data["results"]:
                openalexid = eat_prefix(item["id"])
                title = item.get("title", None)
                abstract = item.get("abstract_inverted_index", None)
                abstract = create_abstract(abstract)
                if title is None:
                    logger.warning(f"Title not found for reference {openalexid}. Skipping this reference.")
                    continue
                self.titles_cache[openalexid] = title
                self.abstracts_cache[openalexid] = abstract
                paper.ref_data.append((title, abstract))

        return self.check_ref_data(paper)

    def check_ref_data(self, paper:Paper):
        if len(paper.ref_data) == 0:
            paper.status = "No valid references found after fetching data."
            return paper
        
        # have we enough abstracts to calculate the score based on abstracts?
        if not paper.titles_only:
            have_abstract = 0
            for _, abstract in paper.ref_data:
                if abstract is not None and isinstance(abstract, str):
                    have_abstract += 1
                    
            if have_abstract < 5:
                paper.titles_only = True
                logger.warning(f"Not enough abstracts found for references of paper {paper.openalexid}. Will calculate score based on titles only.")
                paper.ref_data = [(title, None) for title, _ in paper.ref_data if title is not None]
            else:
                paper.ref_data = [
                    (title, abstract)
                    for title, abstract in paper.ref_data
                    if title is not None and abstract is not None and isinstance(abstract, str)
                ]
        return paper
    
    def fetch_ref_data(self, paper:Paper):
        logger.debug(f"Fetching reference data for paper {paper.openalexid}. Number of references: {len(paper.references)}")
        for ref in paper.references:
            select_fields = []
            # title
            if ref in self.titles_cache:
                title = self.titles_cache[ref]
            else: 
                select_fields.append("title")
            # abstract    
            if not paper.titles_only:
                if ref in self.abstracts_cache:
                    abstract = self.abstracts_cache[ref]
                else:
                    select_fields.append("abstract_inverted_index") 
            else:
                abstract = None

            if not select_fields:
                paper.ref_data.append((title, abstract))
                continue 

            select_str = ",".join(select_fields)
            data = download_paper_data(ref, select=select_str)   
            if data is None:
                logger.warning(f"Error during fetching data for reference {ref}. Skipping this reference.")
                continue
            
            if "title" in data:
                title = data["title"]
                self.titles_cache[ref] = title
            else:
                title = None

            if not paper.titles_only:
                if "abstract" in data: 
                    abstract = data["abstract"]
                    self.abstracts_cache[ref] = abstract
                else:
                    abstract = None
            paper.ref_data.append((title, abstract))

        return self.check_ref_data(paper)
    
    def calculate_embeddings(self, paper:Paper):
        # first try cache 
        if paper.openalexid in self.paper_embeddings_cache:
            paper.embedding = self.paper_embeddings_cache[paper.openalexid]
        if paper.openalexid in self.ref_embeddings_cache:
            paper.ref_embeddings = self.ref_embeddings_cache[paper.openalexid]

        if paper.embedding is None:
            for result in self.embeddings_model.embed(
                [(paper.title, paper.abstract)], 
                titles_only=paper.titles_only
            ):
                pass # we have to skipp all intermediate results
            paper.embedding = result
            self.paper_embeddings_cache[paper.openalexid] = result 

        if paper.ref_embeddings is None:
            for result in self.embeddings_model.embed(
                paper.ref_data,
                titles_only=paper.titles_only
            ):
                pass # we have to skipp all intermediate results
            paper.ref_embeddings = result
            self.ref_embeddings_cache[paper.openalexid] = result 

        if paper.embedding is None or paper.ref_embeddings is None:
            paper.status = "Error during embedding calculation."
            return paper

        return paper

    def return_dummy_scores(self, paper:Paper):
        return {
            "score": -1,
            "n_related": 0,
            "titles_only": paper.titles_only,
            "status": paper.status   
        }

    def eval_paper(self, openalexid:str, title:str=None, abstract:str=None):
        """ Evaluate a paper given its OpenAlex ID. 
        
        Args:
            openalexid (str): The OpenAlex ID of the paper to evaluate.
        
        Returns:
            dict: A dictionary containing the evaluation results.
        """ 
        paper = Paper(openalexid, title, abstract)

        paper = self.fetch_paper_data(paper)
        if paper.status != "OK":
            return self.return_dummy_scores(paper)
        logger.debug(f"Paper data fetched successfully for {openalexid}. Title: {paper.title}, Abstract: {'Yes' if paper.abstract else 'No'}, Number of references: {len(paper.references)}")
        assert isinstance(paper.title, str)
        if not paper.titles_only:
            assert isinstance(paper.abstract, str)
        
        paper = self.fetch_ref_data_batched(paper) 
        logger.debug(f"Paper status after fetching data: {paper.status}.")
        if paper.status != "OK":
            return self.return_dummy_scores(paper)
        logger.debug(f"Reference data fetched successfully for {openalexid}. Number of valid references with data: {len(paper.ref_data)}. Titles only: {paper.titles_only}" )

        paper = self.calculate_embeddings(paper)
        if paper.status != "OK":
            return self.return_dummy_scores(paper)
        logger.debug(f"Embeddings calculated successfully for {openalexid}.")

        score = Score()
        
        result = {}
        result["paper_ref"] = score.eval_paper_ref_dissimilarity(paper)
        result["ref_ref"] = score.eval_ref_ref_dissimilarity(paper)
        result["ref_spread"] = score.eval_ref_spread(paper)  

        result["n_related"] = len(paper.ref_data)
        result["titles_only"] = paper.titles_only
        result["status"] = paper.status

        if not self.online:
            self.save_cache() 

        return result
