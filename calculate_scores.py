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

@click.command()
@click.argument("filename", type=click.Path(exists=True), default="data/challenge_data.csv")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), 
              default="INFO")
def main(filename, log_level):
    result_backup = "data/challenge_scores.pickle"
    result_filename = "data/challenge_scores_final.csv" 
    id_string = "OpenAlexID (as URL)"
    paper_id = "PaperProjectID" 
    title_string = "Title"
    abstract_string = "Abstract"
    logger.setLevel(log_level.upper())


    logger.info(f"Starting score calculation script... reading input file {filename}")
    df = pd.read_csv(filename)
    logging.info(f"Read {len(df)} rows from input file.") 

    evaluator = Evaluator(online=False) 
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
        scores = evaluator.eval_paper(openalex_id, 
                                      title=row.get(title_string, None), 
                                      abstract=row.get(abstract_string, None))
        print(f"Row {index} - OpenAlex ID: {openalex_id} - Scores: {scores}")
        results[pid] = scores 
        
        if index % 10 == 0:
            with open(result_backup, "wb") as f:
                pickle.dump(results, f)
            logging.info(f"Saved intermediate results to {result_backup} after processing {index} rows.")

if __name__ == "__main__":
    main()

