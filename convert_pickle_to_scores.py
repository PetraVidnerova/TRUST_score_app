import click 
import pickle
import pandas as pd


@click.command()
@click.argument("pickled_file", default="data/challenge_scores.pickle")
@click.option("--output-file", default="data/challenge_final_scores.csv")
def main(pickled_file, output_file):

    with open(pickled_file, "rb") as f:
        scores = pickle.load(f)
    scores_df = pd.DataFrame.from_dict(scores, orient="index").sort_index()

    min_max = {
        "paper_ref": {"min":  0.0, "max": 0.223247},
        "ref_ref": {"min": 0.0, "max": 0.193948}
    }
    
    def normalize(col):
        min = min_max[col.name]["min"]
        max = min_max[col.name]["max"]
        return (col - min) / (max - min)

    scores_df["trust_score"] = normalize(scores_df["paper_ref"]).combine(
        normalize(scores_df["ref_ref"]), max)

    def convert(col):
        # convert to 1-100 scale frm 0-1
        return 1.0 + col * 99.0
    
    scores_df["converted_score"] = convert(scores_df["trust_score"])  

    scores_df[["paper_ref", "ref_ref", "trust_score", "converted_score"]].to_csv(output_file)
    print(f"Final scores saved to {output_file}")

if __name__ == "__main__":
    main()

