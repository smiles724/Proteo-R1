import glob
import json
import os
from os.path import join, dirname, basename

import protein_eval
from protein_eval.utils.table import list_to_table


RESULT_ROOT = join(dirname(protein_eval.__file__), "../..", 'eval_results')


def main():
    model_list = []
    for folder in os.listdir(RESULT_ROOT):
        if os.path.isdir(join(RESULT_ROOT, folder)):
            model_list.append(folder)
    model_list.sort()

    print_score_list = []
    for model_name in model_list:
        model_path = join(RESULT_ROOT, model_name)
        json_result_list = glob.glob(join(model_path, '**/*.json'), recursive=True)
        json_result_list.sort()

        for json_result_path in sorted(json_result_list):
            print("[DEBUG]", json_result_path)
            sampling_params_name = basename(dirname(json_result_path))

            with open(json_result_path, 'r') as f:
                json_result = json.load(f)

            acc_list = []
            first_num = 0
            second_num = 0
            error_num = 0
            for doc in json_result:
                doc_gt = doc["gt"]
                doc_parsed_ans = doc["parsed_ans"]
                doc_parsed_lvl = doc.get("parsed_levels", [None] * len(doc_parsed_ans))
                for ans, lvl in zip(doc_parsed_ans, doc_parsed_lvl):
                    if ans is None:
                        acc_list.append(0)
                        error_num += 1
                        continue

                    if ans.strip().lower() == doc_gt.strip().lower():
                        acc_list.append(1)
                    else:
                        acc_list.append(0)

                    if lvl == "first" or lvl is None:
                        first_num += 1
                    elif lvl == "second":
                        second_num += 1
                    elif lvl == "error":
                        error_num += 1
                    else:
                        raise ValueError("")

            avg_acc = sum(acc_list) / len(acc_list)
            print_score_list.append([model_name, sampling_params_name, f"{avg_acc:.2%}", len(acc_list), first_num, second_num, error_num])

    custom_headers = ["model", "sampling_params", "avg_acc", "data_num", "first_num", "second_num", "error_num"]
    print_table = list_to_table(print_score_list, custom_headers)
    score_save_path = f"{RESULT_ROOT}/score_table.txt"
    os.makedirs(os.path.dirname(score_save_path), exist_ok=True)
    with open(score_save_path, "w") as f:
        f.write(str(print_table))
    print(print_table)


if __name__ == '__main__':
    main()
