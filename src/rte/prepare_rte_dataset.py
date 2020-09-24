import os

from copy import deepcopy
from typing import List, Union, Dict, Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import json
import os
import csv
import numpy as np
from unicodedata import normalize
from collections import defaultdict
from random import shuffle
from tqdm import tqdm
import argparse
import logging
import sys
import json
import numpy as np
import json

# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def process_data(args):
    # path to wiki-dump files
    path_to_data = args.wiki_path
    # collect the names of all wiki files
    json_files = [fname for fname in os.listdir(path_to_data) if fname.startswith('wiki-') and fname.endswith('.jsonl')]

    def choice_from_list(some_list, sample_size):
        """
        Input: a list and desired sample size
        Returns: a random sample of the desired size, the same list with this sample removed
        """
        res = np.random.choice(some_list, sample_size, replace = False)
        new_list = [x for x in some_list if x not in res]
        return res, new_list

    # compile wiki files into one massive dictionary
    wiki_data = {}
    for i,f in enumerate(json_files):
        with open(os.path.join(path_to_data, f)) as jreader:
            for itm in jreader:
                j = json.loads(itm)
                wiki_data[normalize('NFC', j['id'])] = j['lines'].split('\n')
            print("Parsing {} of {} data chunks. Total entries: {}".format(i+1, len(json_files), len(wiki_data)))


    # create csv to write to (split into train, test, eval sets)
    with open(args.out_file, 'w+') as output_file:
        out_data = csv.writer(output_file)
        
        out_data.writerow(['id', 'verifiable', 'label', 'claim', 'evidence'])

        # collect evidence ids and pair to wiki_data based on sentence index
        evidence_dict = defaultdict(dict)
        with open(args.in_file) as jreader:
            for itm in jreader:
                j = json.loads(itm)
                id = str(j['id'])
                evidence_dict[id] = {}

                # change 'verifiable' and 'label' into integers for easier manipulation
                if j['verifiable'] == 'VERIFIABLE':
                    verifiable = 1
                else:
                    verifiable = 0
                if j['label'] == 'SUPPORTS':
                    label = 1
                elif j['label'] == 'REFUTES':
                    label = 2
                else:
                    label = 0

                # build evidence_dict based on current element of json file
                evidence_dict[id]['verifiable'] = verifiable
                evidence_dict[id]['label'] = label
                evidence_dict[id]['claim'] = j['claim']

                # initialize each id's evidence as an empty list
                evidence_dict[id]['evidence'] = []

                # add all evidence pieces
                if label == 0:
                    print ("NEI", id, len(j['predicted_sentences']))
                    for pred_sent in j['predicted_sentences']:
                        article_name = pred_sent[0]
                        sentence_id = pred_sent[1]
                        if sentence_id is not None:
                            try:
                                article_name = normalize('NFC', article_name)
                                current_sentence = wiki_data[article_name][sentence_id].split('\t')[1]
                                if current_sentence not in evidence_dict[id]['evidence']:
                                    evidence_dict[id]['evidence'].append(current_sentence)
                            except KeyError as e:
                                print(article_name, ' is not in available evidence.')
                                pass
                else:
                    for e in j['evidence']:
                        for evidence in e:
                            anno_id = evidence[0]
                            evidence_id = evidence[1]
                            article_name = evidence[2]
                            sentence_id = evidence[3]
                            if sentence_id is not None:
                                try:
                                    article_name = normalize('NFC', article_name)
                                    current_sentence = wiki_data[article_name][sentence_id].split('\t')[1]
                                    if current_sentence not in evidence_dict[id]['evidence']:
                                        evidence_dict[id]['evidence'].append(current_sentence)
                                except KeyError as e:
                                    print(article_name, ' is not in available evidence.')
                                    pass

        # # dictionaries separating data into their categories (supports, refutes, nei)
        # supports_dict = {}
        # refutes_dict = {}
        # nei_dict = {}

        # # sort data into supports/refutes/nei
        # for key, data in evidence_dict.items():
        #     if data['label'] == 1:
        #         supports_dict[key] = data
        #     elif data['label'] == 2:
        #         refutes_dict[key] = data
        #     else:
        #         nei_dict[key] = data

        # # compile lists of ids for support, refute, nei
        # support_keys = list(supports_dict.keys())
        # refute_keys = list(refutes_dict.keys())
        # nei_keys = list(nei_dict.keys())

        # # separate data into test, eval, and train (eval and test should each have 14500 data entries; train gets the rest)
        # n = 14500 // 3
        # test_support_keys, support_keys = choice_from_list(support_keys, n)
        # eval_support_keys, train_support_keys = choice_from_list(support_keys, n)
        # test_refute_keys, refute_keys = choice_from_list(refute_keys, n)
        # eval_refute_keys, train_refute_keys = choice_from_list(refute_keys, n)
        # test_nei_keys, nei_keys = choice_from_list(nei_keys, n)
        # eval_nei_keys, train_nei_keys = choice_from_list(nei_keys, n)

        # # add together support, refute, and nei keys, then shuffle them for randomization
        # train_keys = np.concatenate((train_support_keys, train_refute_keys, train_nei_keys))
        # eval_keys = np.concatenate((eval_support_keys, eval_refute_keys, eval_nei_keys))
        # test_keys = np.concatenate((test_support_keys, test_refute_keys, test_nei_keys))
        # shuffle(train_keys)
        # shuffle(eval_keys)
        # shuffle(test_keys)

        # write to each csv file
        # files = [(train_keys, train_data), (eval_keys, eval_data), (test_keys, test_data)]
        # for keys, file in files:
        #     for k in keys:
        #         try:
        #             file.writerow([k, evidence_dict[k]['verifiable'], evidence_dict[k]['label'], evidence_dict[k]['claim'], evidence_dict[k]['evidence']])
        #         except KeyError as e:
        #             print(e)
        #             raise

        for key, data in evidence_dict.items():
            try:
                out_data.writerow([key, data['verifiable'], data['label'], data['claim'], data['evidence']])
            except KeyError as e:
                print(e)
                raise
        

if __name__ == "__main__":
    # LogHelper.setup()
    # LogHelper.get_logger("allennlp.training.trainer")
    # LogHelper.get_logger(__name__)


    parser = argparse.ArgumentParser()

    parser.add_argument('wiki_path', type=str, help='/path/to/input.jsonl')
    parser.add_argument('in_file', type=str, help='/path/to/input.jsonl')
    parser.add_argument('out_file', type=str, help='/path/to/output.jsonl')



    args = parser.parse_args()
    process_data(args)
