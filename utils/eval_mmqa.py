from evaluator import evaluate_predictions
import json
import os
import pandas as pd
import argparse

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")
SAVE_DIR = os.path.join(ROOT_DIR, "test/files")
# METHOD_SUFFIX = "_oracle_10samples"
# METHOD_SUFFIX = "_6samples_simple"
# METHOD_SUFFIX = "_main_exp_nocot"
# METHOD_SUFFIX = "_ablation_coherent_allcot"
METHOD_SUFFIX = ''

RESULT_PATH = os.path.join(ROOT_DIR,f'results/dpmlb_mmqa_validation{METHOD_SUFFIX}.json')
# RESULT_PATH = os.path.join(ROOT_DIR,f'results/{METHOD_SUFFIX}.json')
# SAVE_TYPES = ["all","ImageQ"]
# SAVE_TYPES = ["all"]
SAVE_TYPES = []

CONFIG = {
    "Method ID": 1,
    "Prompt Type": "Simple",
    "ImageCaption": "LLaVA",
    "SQL PostProcess": "no",
    "n_samples": 6,
    "Execute n-shot": 5
}



ALL_QUESTION_TYPES = [
    'TextQ',
    'TableQ',
    'ImageQ',
    'ImageListQ',
    'Compose(TableQ,ImageListQ)',
    'Compose(TextQ,ImageListQ)',
    'Compose(ImageQ,TableQ)',
    'Compose(ImageQ,TextQ)',
    'Compose(TextQ,TableQ)',
    'Compose(TableQ,TextQ)',
    'Intersect(TableQ,TextQ)',
    'Intersect(ImageListQ,TableQ)',
    'Intersect(ImageListQ,TextQ)',
    'Compare(Compose(TableQ,ImageQ),TableQ)',
    'Compare(Compose(TableQ,ImageQ),Compose(TableQ,TextQ))',
    'Compare(TableQ,Compose(TableQ,TextQ))',
]

def out_csv(json_data, csv_path):
    df = pd.DataFrame(json_data)
    df.to_csv(csv_path, index=False)


def calc_from_list(array):
    total = 0
    em = 0
    f1 = 0
    for item in array:
        em += item[0] * item[2]
        f1 += item[1] * item[2]
        total += item[2]
    if total == 0:
        return 0, 0
    return (em/total).round(2), (f1/total).round(2)

def eval_by_types(data):
    config = CONFIG.copy()
    imageQ = []
    textQ = []
    single_modalQ = []
    multi_modalQ = []
    for n_type in ALL_QUESTION_TYPES:
        n_data = dict()
        for (eid, d) in data.items():
            if data[eid]['ori_data_item']['type'] == n_type:
                n_data[eid] = d

        if len(n_data) == 0:
            continue
        
        print(f"\n{n_type}: total {len(n_data)} questions\n EM \t F1")
        pred_dict = {eid: [str(n_data[eid]['generations']) if len(str(data[eid]['generations'])) else 'None'] for eid in n_data.keys()}
        gold_dict = {eid: [str(n_data[eid]['ori_data_item']['answer_text'])] for eid in n_data.keys()}
        eval_scores, instance_eval_results = evaluate_predictions(pred_dict, gold_dict)
        EM = (eval_scores['acc']*100).round(2)
        F1 = (eval_scores['f1']*100).round(2)
        config[f'{n_type} EM'] = EM
        config[f'{n_type} F1'] = F1
        print(f"{(eval_scores['acc']*100).round(2)} \t {(eval_scores['f1']*100).round(2)}")
        if n_type in ["ImageQ", "ImageListQ"]:
            imageQ.append((EM,F1,len(n_data)))
        elif n_type in ["TextQ"]:
            textQ.append((EM,F1,len(n_data)))

        if n_type in ["TableQ", "ImageQ", "TextQ", "ImageListQ"]:
            single_modalQ.append((EM,F1,len(n_data)))
        else:
            multi_modalQ.append((EM,F1,len(n_data)))
        if n_type in SAVE_TYPES:
            with open(os.path.join(SAVE_DIR,f"pred_dict_{n_type}.json"), "w") as f:
                json.dump(pred_dict, f, indent=4)
            with open(os.path.join(SAVE_DIR,f"gold_dict_{n_type}.json"), "w") as f:
                json.dump(gold_dict, f, indent=4)
    print("\n\n")
    print("ImageQ: ", calc_from_list(imageQ))
    print("TextQ: ", calc_from_list(textQ))
    print("Single ModalQ: ", calc_from_list(single_modalQ))
    print("Multi ModalQ: ", calc_from_list(multi_modalQ))
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', type=int, default=None)
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()
    right = 0
    all_num = 0
    with open(RESULT_PATH, "r") as f:
        data = json.load(f)

    new_data = dict()
    for eid, d in data.items():
        if type(d['generations'])==type({}):
            continue
        new_data[eid] = d

    data=new_data
    
    if args.l is not None and args.num is not None:
        data = dict(list(data.items())[args.l:(args.l+args.num)])

    result = eval_by_types(data).copy()
    pred_dict = {eid: [str(data[eid]['generations']) if len(str(data[eid]['generations'])) else 'None'] for eid in data.keys()}
    gold_dict = {eid: [str(data[eid]['ori_data_item']['answer_text'])] for eid in data.keys()}

    # for i in range(2441):
        # if str(i) not in pred_dict.keys():
            # print(f'{i} not in pred_dict')
    
    # exit(0)
        
    eval_scores, instance_eval_results = evaluate_predictions(pred_dict, gold_dict)

    result['EM'] = (eval_scores['acc']*100).round(2)
    result['F1'] = (eval_scores['f1']*100).round(2)

    print(f"\n\n Overall: total {len(instance_eval_results)} questions\n EM \t F1")
    print(f"{(eval_scores['acc']*100).round(2)} \t {(eval_scores['f1']*100).round(2)}")

    if args.save:
        out_csv([result], "eval_results.csv")
        with open("eval_results", "w") as f:
            json.dump(instance_eval_results, f, indent=4)
            print("eval_results saved")

    if "all" in SAVE_TYPES:
        with open(os.path.join(SAVE_DIR,"pred_dict_all.json"), "w") as f:
            json.dump(pred_dict, f, indent=4)
        with open(os.path.join(SAVE_DIR,"gold_dict_all.json"), "w") as f:
            json.dump(gold_dict, f, indent=4)