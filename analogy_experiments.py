import polars as pl
import json

def preprocess_test_set(df_path:str, output_path="experiments/inference/"):
    df = pl.read_json(df_path)
    # Get unique sentences
    df_sentence = df.unique(subset="sentence_id")
    # Get serialised sentences to keep track on later
    df_sentence = df_sentence.with_columns(pl.Series([i for i in range(df_sentence.shape[0])]).alias("parser_id"))
    # Get the python strings ready
    sentences = []
    for item in df_sentence.iter_rows(named=True):
        instance = {}
        instance["sentence"] = item["text"]
        sentences.append(json.dumps(instance))
    with open(f"{output_path}test.json", "w") as file:
        file.write("\n".join(sentences))
    return "\n".join(sentences)

output = preprocess_test_set("data/test.json")
