import os
from collections import OrderedDict
import argparse
from datasets import load_dataset
from mlb_utils import get_all_paragraph_plans
type_map = {"train": "train", "test": "test", "valid": "validation"}


def process(type, output_folder, suffix):
    mlb_dataset = load_dataset("GEM/mlb_data_to_text")
    output_file = open(os.path.join(output_folder, type + "." + suffix + ".pp"), mode="w", encoding="utf-8")
    data = mlb_dataset[type_map[type]]
    for entry_index, entry in enumerate(data):
        output = get_all_paragraph_plans(entry, entry_index)

        descs_list = list(OrderedDict.fromkeys(output))
        prefix_tokens_=  ["<unk>", "<blank>", "<s>", "</s>", "<end-plan>", "<empty-segment>"]
        input_template = " <segment> " + " <segment> ".join(descs_list)
        input_template = " ".join(prefix_tokens_) + input_template
        output_file.write(input_template)
        output_file.write("\n")

        if entry_index % 50 == 0:
            print("entry_index", entry_index)
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating inference plans for mlb dataset')
    parser.add_argument('-output_folder', type=str,
                        help='path of output file', default=None)
    parser.add_argument('-dataset_type', type=str,
                        help='type of dataset', default=None)
    parser.add_argument('-suffix', type=str,
                        help='suffix', default=None)
    args = parser.parse_args()

    process(args.dataset_type, args.output_folder, args.suffix)
