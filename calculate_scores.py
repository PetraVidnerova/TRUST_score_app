import logging 
import pickle 
from pathlib import Path
import pandas as pd 
import tqdm

from utils.score import Evaluator
from utils.utils import eat_prefix

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s (%(module)s)] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    filename = "data/challenge_data.csv"
    result_backup = "data/challenge_scores.pickle"
    result_filename = "data/challenge_scores_final.csv" 
    id_string = "OpenAlexID (as URL)"
    title_string = "Title"
    abstract_string = "Abstract"
    logger.setLevel(logging.INFO)


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
        logging.debug(f"Evaluating row {index} with OpenAlex ID {row[id_string]}")
        openalex_id = eat_prefix(row[id_string])
        scores = evaluator.eval_paper(openalex_id, 
                                      title=row.get(title_string, None), 
                                      abstract=row.get(abstract_string, None))
        print(f"Row {index} - OpenAlex ID: {openalex_id} - Scores: {scores}")

        if index % 10 == 0:
            with open(result_backup, "wb") as f:
                pickle.dump(results, f)
            logging.info(f"Saved intermediate results to {result_backup} after processing {index} rows.")

if __name__ == "__main__":
    main()

