from ..fewshot_base import FewshotTask
import os
import json
import re
import random
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable, Iterable
from ..registry import task_register

from worldsense.benchmark import load_testset

@task_register("worldsense")
class WorldSenseFewshot(FewshotTask):
    # load data directly
    script_path = os.path.abspath(__file__)
    try:
        main_test_set = load_testset("./data/test_set")
        other_tests = dict()
        for test in os.listdir("./data/other_tests"):
            other_tests[test] = load_testset(f"./data/other_tests/{test}")
    except:
        original_data = load_testset("./data/downstream_data/worldsense/test_set")

    @classmethod
    def build_task(cls, *args, **kwargs):
        if kwargs['train_path'] is None:
            kwargs['train_path'] = "./data/downstream_data/worldsense/training_set/trials_100k.jsonl.bz2"
        if kwargs['test_path'] is None:
            kwargs['test_path'] = "./data/downstream_data/worldsense/test_set"
        task = cls(*args, **kwargs)
        return task

    def __init__(self, train_path=None, test_path=None, **kwargs):
        # loop_inference_num_examples = range(0, 17)
        loop_inference_num_examples = kwargs.get("loop_inference_num_examples", [5, 0])
        use_logits = kwargs.get("use_logits", False)
        generation_kwargs = {
            "use_cache": True,
            "max_new_tokens": 2 if use_logits else 512,
            "do_sample": False,
            "top_k": 0,
            "top_p": 0.9,
            "temperature": 0.7,
            "repetition_penalty": 1.05,
            "return_dict_in_generate": use_logits,
            "output_scores": use_logits
        }

        self.choices = None

        labels_map = None
        task_desc_prompt = ""
        super().__init__(train_path=train_path, test_path=test_path,task_name="worldsense",
                         task_desc_prompt=task_desc_prompt, labels_map=labels_map,
                         loop_inference_num_examples=loop_inference_num_examples,
                         generation_kwargs=generation_kwargs,
                         choices=self.choices, **kwargs)
        
    def get_data(self, train_path, test_path):
        if self.main_test_set is not None:
            self.test_data = self.main_test_set['text'].tolist()
        else:
            self.test_data = load_testset(test_path)['text'].tolist()
        random.seed(self.seed)

    def get_label(self, example) -> str:
        return example['goldresp']
    
    def create_fewshot_prompt(self, item: Dict, num_examples=0, train_data=None, examples_strategy="random", **kwargs):
        def split(a, b):
            base, res = a // b, a % b
            boxes = [base for i in range(b)]
            res_ids = random.sample(range(b), res)
            for i in res_ids:
                boxes[i] += 1
            return boxes
        def sample_data(df, num):
            format_data = lambda x : f"Q: {x['text']}\nA: {x['goldresp']}"
            return map(format_data, random.sample(df.to_dict(orient='records'), num))
        
        inputs = item.get('text')
        if num_examples == 0:
            return f"Q: {inputs}\nA: "
        
        examples = []
        split_num = split(num_examples, 4)
        for each_split in split_num:
            examples.extend(sample_data(train_data, each_split))
        
        return f"Follow the given examples and answer the question. Let's think step by step.\n{'\n'.join(examples)}\n\nQ: {inputs}\nA:."
    
    # 此数据集的gold response为单选
    # all(map(lambda x : x.isdigit() or x.isalpha(), test_set['goldresp'])) equals True
    def _postprocess(self, output_str, item: Dict = None):
        if item is not None:
            expectedresp = item['expectedresp']
            goldresp = item['goldresp']
            falseresp = list(set.difference(set(expectedresp), set(goldresp)))
            lower_goldresp = goldresp.lower()
            if lower_goldresp == output_str.lower():
                return goldresp
            elif lower_goldresp in output_str.lower():
                if any(map(lambda x: x in output_str.lower(), falseresp)):
                    return ''
                return goldresp
   
    def not_follow_instruct(self, example, ans):
        return False
    
    def inference(self, dataloader, model, tokenizer, num_examples, print_log=True, strategy=None, generation_kwargs=None, **kwargs):
        #if num_examples != 3 and num_examples != 0:
        #    print("[warn]: num_examples only support 3 or 0!")
        #    if num_examples >0 and num_examples !=3:
        #        num_examples = 3
        return super().inference(dataloader, model, tokenizer, num_examples, print_log, strategy = None, generation_kwargs = generation_kwargs, **kwargs)
