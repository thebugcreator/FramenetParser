import polars as pl
import torch

from scipy.spatial.distance import cdist

from tqdm import tqdm


def get_distances(target_vects, source_vects, metric):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
    return cdist(target_vects, source_vects, metric=metric).tolist()


def get_ref_sentence(target_text, frame_id, dataset, metric="cosine", model_path="sentence-transformers/LaBSE", top_n=10, n_out=1):
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_path)
    target_embs = encoder.encode(target_text)
    df_filtered = dataset.filter(pl.col("frame_id")==frame_id)
    source_stns = df_filtered.select("text").to_series()
    if not source_stns.is_empty():
        source_embs = [encoder.encode(item) for item in tqdm(source_stns, desc=f"Encoding {source_stns.shape[0]} candidate(s)")]
        distances   = get_distances([target_embs], source_embs, metric)[0]
        df_filtered = df_filtered.with_columns(pl.Series(distances).alias("distance")).sort(by="distance")
        num_roles   = [len(item.split("#")) for item in df_filtered["fe_idx"]]
        df_filtered = df_filtered.with_columns(pl.Series(num_roles).alias("n_roles"))
        output_stn  = df_filtered.head(top_n).sort(by="n_roles", descending=True).head(n_out)
        return output_stn
    else:
        raise Exception("No reference in the dataset")
