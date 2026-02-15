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


    scores_df.to_csv(output_file)
    print(f"Final scores saved to {output_file}")

if __name__ == "__main__":
    main()

