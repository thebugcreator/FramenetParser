from nltk.corpus import framenet as fn

import polars as pl

import argparse
from os import listdir
from os.path import isfile, join
import shutil


TEST_FILES = [
        "ANC__110CYL067.xml",
        "ANC__110CYL069.xml",
        "ANC__112C-L013.xml",
        "ANC__IntroHongKong.xml",
        "ANC__StephanopoulosCrimes.xml",
        "ANC__WhereToHongKong.xml",
        "KBEval__atm.xml",
        "KBEval__Brandeis.xml",
        "KBEval__cycorp.xml",
        "KBEval__parc.xml",
        "KBEval__Stanford.xml",
        "KBEval__utd-icsi.xml",
        "LUCorpus-v0.3__20000410_nyt-NEW.xml",
        "LUCorpus-v0.3__AFGP-2002-602187-Trans.xml",
        "LUCorpus-v0.3__enron-thread-159550.xml",
        "LUCorpus-v0.3__IZ-060316-01-Trans-1.xml",
        "LUCorpus-v0.3__SNO-525.xml",
        "LUCorpus-v0.3__sw2025-ms98-a-trans.ascii-1-NEW.xml",
        "Miscellaneous__Hound-Ch14.xml",
        "Miscellaneous__SadatAssassination.xml",
        "NTI__NorthKorea_Introduction.xml",
        "NTI__Syria_NuclearOverview.xml",
        "PropBank__AetnaLifeAndCasualty.xml",
        ]

DEV_FILES = [
        "ANC__110CYL072.xml",
        "KBEval__MIT.xml",
        "LUCorpus-v0.3__20000415_apw_eng-NEW.xml",
        "LUCorpus-v0.3__ENRON-pearson-email-25jul02.xml",
        "Miscellaneous__Hijack.xml",
        "NTI__NorthKorea_NuclearOverview.xml",
        "NTI__WMDNews_062606.xml",
        "PropBank__TicketSplitting.xml",
        ]


def check_path(path:str):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)

def clean_dir(path:str):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    for file in Path(path).glob("*"):
        if file.is_file():
            file.unlink()

def prepare_data_from_root():
    path_fulltext   = "data/fndata-1.7/framenet_v17/fulltext/"
    path_dataset    = "data/fndata-1.7/"
    path_ds_train   = path_dataset + "train/fulltext/"
    path_ds_dev     = path_dataset + "dev/fulltext/"
    path_ds_test    = path_dataset + "test/fulltext/"
    check_dir       = clean_dir # For future usage: refresh the folder or just create it 
    check_dir(path_ds_train)
    check_dir(path_ds_dev)
    check_dir(path_ds_test)

    all_files       = [f for f in listdir(path_fulltext) if isfile(join(path_fulltext, f))]

    for file in all_files:
        if ".xml" in file:
            if file in TEST_FILES:
                shutil.copyfile(path_fulltext + file, path_ds_test + file)
            elif file in DEV_FILES:
                shutil.copyfile(path_fulltext + file, path_ds_dev + file)
            else:
                shutil.copyfile(path_fulltext + file, path_ds_train + file)
    return all_files

def get_alien_dataset(write_json=True, json_path="data/fndata-1.7/alien/sentences.json"):
    path_fulltext   = "data/fndata-1.7/framenet_v17/fulltext/"
    all_xml_files   = [f for f in listdir(path_fulltext) if (isfile(join(path_fulltext, f)) and ".xml" in f)]
    covered_docs    = {item.replace(".xml", "").split("__")[1] for item in all_xml_files}

    fndocs          = {doc.name:doc.ID for doc in fn.docs()}
    covered_docids  = {fndocs[cdocname] for cdocname in covered_docs}

    alien_sentences = [sent for sent in fn.sents() if ("docID" not in sent.keys() or sent.docID not in covered_docids)]
    
    if write_json:
        alien_records   = [{"ID":sent.ID,"text":sent.text, "frameID":sent.frame.ID} for sent in alien_sentences]
        df_alien_sents  = pl.DataFrame(alien_records)
        df_alien_sents.write_json(json_path)
    return alien_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing if lazy")
    parser.add_argument("function", type=int, default=0, help="Choose the functionality")

    args = parser.parse_args()

    arg_function    = args.function
    if arg_function == 0:
        prepare_data_from_root()
    elif arg_function   == 1:
        export_dir  = "data/fndata-1.7/alien/"
        clean_dir(export_dir)
        all_sentences   = get_alien_dataset(write_json=True, json_path=export_dir+"sentences.json")
    
        
    
