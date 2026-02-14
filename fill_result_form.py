import click 
import pickle
import pandas as pd


@click.command()
@click.argument("pickled_file", default="challenge/challenge_scores.pickle")
@click.option("--template-file", default="challenge/Challenger_Response_Form.csv", help="Path to the response form template.")
@click.option("--output-file", default="challenge/Challenger_Response_Form_filled.csv", help="Path to save the filled response form.")
def main(pickled_file, template_file, output_file):
    template = pd.read_csv(template_file, index_col=0)

    with open(pickled_file, "rb") as f:
        scores = pickle.load(f)
    scores_df = pd.DataFrame.from_dict(scores, orient="index").sort_index()


    def normalize(col):
        min = col.min()
        max = col.max()
        return (col - min) / (max - min)

    scores_df["raw_score"] = normalize(scores_df["paper_ref"]).combine(
        normalize(scores_df["ref_ref"]), max)


    # fill template with the scores
    template["NoveltyRaw"] = scores_df["raw_score"]

    not_computed = scores_df["score"] == -1
    title_only = scores_df["titles_only"] == True 

    template.loc[not_computed, "NoveltyRaw"] = -1

    template["NoveltyRaw_Certainty"] = 1.0
    template.loc[not_computed, "NoveltyRaw_Certainty"] = 0.0
    template.loc[title_only, "NoveltyRaw_Certainty"] = 0.5   

    # convert raw  
    template["NoveltyConverted"] = template["NoveltyRaw"]
    template.loc[not_computed, "NoveltyConverted"] = 0.5 

    template["NoveltyConverted_Certainty"] = 1.0
    template.loc[not_computed, "NoveltyConverted_Certainty"] = 0.0
    template.loc[title_only, "NoveltyConverted_Certainty"] = 0.5

    # binary novelty 
    template["NoveltyBinary"] = (template["NoveltyConverted"] >= 0.675).astype(float)
    template["NoveltyBinary_Certainty"] = 1.0
    template.loc[not_computed, "NoveltyBinary_Certainty"] = 0.0
    template.loc[title_only, "NoveltyBinary_Certainty"] = 0.5

    # rank the papers by novelty
    template["Rank"] = template["NoveltyRaw"].rank(ascending=False, method="max")

    # save the filled template
    template.to_csv(output_file)

if __name__ == "__main__":
    main()

