import polars as pl
import json
import re

def find_substring_index(input:str, substr:str):
    return tuple(((match.start(), match.end()) for match in re.finditer(substr, input)))


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

def process_eval_result(result_path:str="output_test.json", reference_path:str="experiments/inference/test_reference.json") -> pl.DataFrame:
    df_reference = pl.read_json(reference_path)
    json_lines = []
    # Read the json result file
    with open(result_path) as file:
        for line in file:
            json_line = json.loads(line)
            json_lines.append(json_line)
    # Process the results record by record
    results = []
    for parser_id, record in enumerate(json_lines):
        tokens = record["words"]
        frame_elements = record["frame_elements"]
        sentence_id = df_reference[parser_id]["sentence_id"][0]
        original_text = df_reference[parser_id]["text"][0]
        for predicate, (onset, offset), role_name in frame_elements:
            pred_onset, pred_offset = predicate[0][0]
            frame_name = predicate[1]
            target_tokens = "#".join(tokens[pred_onset: pred_offset+1])
            element_tokens = "#".join(tokens[onset:offset+1])
            results.append(
                {
                    "sentence_id": sentence_id,
                    "frame_name": frame_name, 
                    "target_idx": f"{pred_onset},{pred_offset+1}",
                    "target_tokens": target_tokens,
                    "text": original_text,
                    "fe_idx": f"{onset},{offset+1}",
                    "fe_role": role_name,
                    "element_tokens": element_tokens,
                }
                )
    df_results = pl.DataFrame(results)
    df_results_agg = df_results.group_by(["sentence_id","frame_name", "target_idx", "target_tokens"]).agg(df_results.columns[4:])
    
    return df_results, df_results_agg

