import logging 
import pickle 
from pathlib import Path

import click 
import pandas as pd 
import tqdm

from utils.score import Evaluator
from utils.utils import eat_prefix

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s (%(module)s)] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

@click.command()
@click.argument("filename", type=click.Path(exists=True), default="data/challenge_data.csv")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), 
              default="INFO")
@click.option("--use-api-key", is_flag=True, default=False)
@click.option("--force-cpu", is_flag=True, default=False)
@click.option("--only-cached", is_flag=True, default=False, help="Only evaluate papers that have cached data available, skip others.")
def main(filename, log_level, use_api_key, force_cpu, only_cached):
    result_backup = "data/challenge_scores.pickle"
    result_filename = "data/challenge_scores_final.csv" 
    id_string = "OpenAlexID (as URL)"
    paper_id = "PaperProjectID" 
    title_string = "Title"
    abstract_string = "Abstract"
    logger.setLevel(log_level.upper())

    if use_api_key:
        with open("openalex_api_key.txt", "r") as f:
            api_key = f.read().strip()
    else:
        api_key = None

    logger.info(f"Starting score calculation script... reading input file {filename}")
    df = pd.read_csv(filename)
    logging.info(f"Read {len(df)} rows from input file.") 

    evaluator = Evaluator(online=False, api_key=api_key, force_cpu=force_cpu,
                          only_cached=only_cached) 
    logging.debug("Evaluator initialized.")

    if Path(result_backup).exists():
        with open(result_backup, "rb") as f:
            results = pickle.load(f)
        logging.info(f"Loaded existing results from {result_backup}, containing {len(results)} entries.")
    else:
        results = {}


    for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        pid = row[paper_id]
        if pid in results:
            continue
        logging.debug(f"Evaluating row {index} with OpenAlex ID {row[id_string]}")
        openalex_id = eat_prefix(row[id_string])
        title = row.get(title_string, None)
        if title is not None and not isinstance(title, str):
            title = None
        abstract = row.get(abstract_string, None)
        if abstract is not None and not isinstance(abstract, str):
            abstract = None
        scores = evaluator.eval_paper(openalex_id, 
                                      title=title, 
                                      abstract=abstract)
        print(f"Row {index} - OpenAlex ID: {openalex_id} - Scores: {scores}")
        results[pid] = scores 
        
        if index % 10 == 0:
            with open(result_backup, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Saved intermediate results to {result_backup} after processing {index} rows.")
    with open(result_backup, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Results saved into {result_backup}. Game over.")
            
if __name__ == "__main__":
    main()

