import click 
import pickle
import pandas as pd


@click.command()
@click.argument("pickled_file", default="data/challenge_scores.pickle")
@click.option("--template-file", default="data/Challenger_Response_Form.csv", help="Path to the response form template.")
@click.option("--output-file", default="data/Challenger_Response_Form_filled.csv", help="Path to save the filled response form.")
def main(pickled_file, template_file, output_file):
    template = pd.read_csv(template_file, index_col=0)

    helper = pd.DataFrame()
    helper["PaperProjectID"] = template.index
    
    with open(pickled_file, "rb") as f:
        scores = pickle.load(f)
    scores_df = pd.DataFrame.from_dict(scores, orient="index").sort_index()
    
    # add also rows for missing IDs 
    original_columns = scores_df.columns.to_list()
    scores_df = (
        pd.merge(template, scores_df, 
                 left_on=template.index,
                 right_on=scores_df.index, 
                 how="left")
        .rename(columns={"key_0": "PaperProjectID"})
        .set_index("PaperProjectID")
    )[original_columns]
    
    min_max = {
        "paper_ref": {"min":  0.0, "max": 0.223247},
        "ref_ref": {"min": 0.0, "max": 0.193948}
    }
    
    def normalize(col):
        min = min_max[col.name]["min"]
        max = min_max[col.name]["max"]
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

