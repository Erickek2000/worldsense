from ..fewshot_base import FewshotTask
import os
import json
import re
import random
import pandas as pd
from typing import Optional, List, Dict, Tuple, Callable, Iterable
from ..registry import task_register

from .worldsense_source.benchmark import (
    load_testset,
    load_trainset)

@task_register("worldsense")
class WorldSenseFewshot(FewshotTask):
    # locate the data
    script_path = os.path.abspath(__file__)
    task_code_base = os.path.dirname(script_path)
    path_main_trainset = os.path.join(task_code_base, "data/training_set/trials_100k.jsonl.bz2")
    path_testset = os.path.join(task_code_base, "data/test_set")
    path_other_testsets = os.path.join(task_code_base, "data/other_tests")
    # load the data
    datasets = dict()
    main_training_set = load_trainset(path_main_trainset); datasets['main_training_set'] = main_training_set
    main_test_set = load_testset(path_testset); datasets['main_test_set'] = main_test_set
    for test in os.listdir(os.path.join(task_code_base, "data/other_tests")):
        datasets[test] = load_testset(os.path.join(task_code_base, "data/other_tests")+f"/{test}")
    # except:
    #    original_data = load_testset("./data/# downstream_data/worldsense/test_set")

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
            "temperature": 0.0,
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
        self.test_data = self.main_test_set.to_dict(orient='records')

    def get_label(self, example) -> str:
        return example['goldresp']
    
    def create_fewshot_prompt(self, item: Dict, num_examples=0, train_data=None, examples_strategy="random", **kwargs):
        # split the num_examples into 5 parts
        # main_train : all_other_tests (4 parts) {\approx} 1 : 1
        def split(num_examples):
            base0, res0 = divmod(num_examples, 2)
            base1, res1 = divmod(base0, 4)
            boxes = [(base1 if i >= 1 else base0) for i in range(5)]
            res_ids = [0] + random.sample(range(1, 5), res1)
            for id in res_ids:
                boxes[id] += ( 1 if id >=1 else res0 )
            return boxes
        # format the data into the string
        # sample data from the list(dicts)
        def sample_data(set_name, num):
            format_data = (
                (lambda x : f"Q: {x['text']}\nA: {x['goldresp']}") if set_name != 'main_training_set' else 
                (lambda x : f"Q: {x['dialog_history']['messages'][0]['content']}\nA: {x['target_message']}")
            )
            return map(format_data, random.sample(self.datasets[set_name].to_dict(orient='records'), num))
        
        inputs = item.get('text')
        if num_examples == 0:
            return f"Q: {inputs}\nA: "
        elif num_examples <= 1:
            raise ValueError("num_examples must be greater than 1")
        
        examples = []
        splits = split(num_examples=num_examples)
        for each_split, set in zip(splits, filter(lambda x : x != 'main_test_set', self.datasets.keys())):
            examples.extend(sample_data(set, each_split))
        hints = '\n\n'.join(examples)
        return f"Follow the given examples and answer the question. Let's think step by step.\n{hints}\n\nQ: {inputs}\nA:."
    
    # 此数据集为单选，gold resp 为正确答案，expected resp 为所有可能的答案
    # all(map(lambda x : x.isdigit() or x.isalpha(), test_set['goldresp'])) {\equals} True
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
            else:
                return ''
        else:
            return ''
   
    def not_follow_instruct(self, example, ans):
        return False
    
    def inference(self, dataloader, model, tokenizer, num_examples, print_log=True, strategy=None, generation_kwargs=None, **kwargs):
        #if num_examples != 3 and num_examples != 0:
        #    print("[warn]: num_examples only support 3 or 0!")
        #    if num_examples >0 and num_examples !=3:
        #        num_examples = 3
        return super().inference(dataloader, model, tokenizer, num_examples, print_log, strategy = None, generation_kwargs = generation_kwargs, **kwargs)
