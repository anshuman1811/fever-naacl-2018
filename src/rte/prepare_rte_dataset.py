import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from allennlp.common import Params
from allennlp.common.tee_logger import TeeLogger
#from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary, Dataset, DataIterator, DatasetReader, Tokenizer, TokenIndexer
from allennlp.models import Model, archive_model, load_archive
from allennlp.service.predictors import Predictor
from allennlp.training import Trainer
from common.util.log_helper import LogHelper
from retrieval.fever_doc_db import FeverDocDB
from rte.parikh.reader import FEVERReader
from tqdm import tqdm
import argparse
import logging
import sys
import json
import numpy as np
import json

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def process_data(db: FeverDocDB, args):
    reader = FEVERReader(db,
                                 sentence_level=ds_params.pop("sentence_level",False),
                                 wiki_tokenizer=Tokenizer.from_params(ds_params.pop('wiki_tokenizer', {})),
                                 claim_tokenizer=Tokenizer.from_params(ds_params.pop('claim_tokenizer', {})),
                                 token_indexers=TokenIndexer.dict_from_params(ds_params.pop('token_indexers', {})))

    logger.info("Reading training data from %s", args.in_file)
    data = reader.read(args.in_file).instances

    with open(out_file, 'w') as f:
        json.dump(data, f)
        

if __name__ == "__main__":
    LogHelper.setup()
    LogHelper.get_logger("allennlp.training.trainer")
    LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('db', type=str, help='/path/to/saved/db.db')
    parser.add_argument('in_file', type=str, help='/path/to/input.jsonl')
    parser.add_argument('out_file', type=str, help='/path/to/output.jsonl')



    args = parser.parse_args()
    db = FeverDocDB(args.db)
    process_data(db,args)
