import polars as pl
import json

def preprocess_test_set(df_path:str, output_path="experiments/inference/", write_files=True) -> str:
    df = pl.read_json(df_path)
    # Get unique sentences
    df_sentence = df.unique(subset="sentence_id")
    # Get serialised sentences to keep track on later
    df_sentence = df_sentence.with_columns(pl.Series([i for i in range(df_sentence.shape[0])]).alias("parser_id"))
    # Get the python strings ready
    sentences = []
    for item in df_sentence.select("text").to_series():
        instance = {"sentence":item}
        sentences.append(json.dumps(instance))
    if write_files:
        with open(f"{output_path}test.json", "w") as file:
            file.write("\n".join(sentences))
        df_sentence.write_json(output_path + "test_reference.json")
    return sentences, df_sentence

# def process_eval_result(result_path:str="output_test.json", reference_path:str="experiments/inference/test_reference.json") -> pl.DataFrame:
result_path:str="output_test.json"
reference_path:str="experiments/inference/test_reference.json"
df_reference = pl.read_json(reference_path)
json_lines = []

with open(result_path) as file:
    for line in file:
        json_line = json.loads(line)
        json_lines.append(json_line)
for record in json_lines:
    tokens = record["words"]
    frames = record["frames"]
    frame_elements = record["frame_elements"]
    frelements = list(zip(frames, frame_elements))


