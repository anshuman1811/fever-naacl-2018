from typing import List

from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors import Predictor


@Predictor.register('drwiki-te')
class TextualEntailmentPredictor(Predictor):
    @overrides
    def _batch_json_to_instances(self, json: List[JsonDict]) -> List[Instance]:
        instances = []
        for blob in json:
            instances.extend(self._json_to_instances(blob))
        return instances

    def set_docdb(self,db):
        self.db = db

    def _json_to_instances(self,json):
        hypothesis_text = json["claim"]
        instances = []
        for page,score in json["predicted_pages"]:
            premise_text = self.db.get_doc_text(page)
            instances.append(self._dataset_reader.text_to_instance(premise_text, hypothesis_text))
        return instances
