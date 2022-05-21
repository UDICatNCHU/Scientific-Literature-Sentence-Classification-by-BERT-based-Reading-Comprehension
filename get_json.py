import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm
import os
import argparse


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--dataset", default='pm20k', type=str, required=True, help="pm20k, pm200k, or nicta.")
    parser.add_argument("--dataset_dir", default='data/PubMed_20k_RCT/', type=str, required=True, help="Path to the dataset dir.")
    parser.add_argument("--file_type", default='train', type=str, required=True, help='train, test, or dev.')
    parser.add_argument("--max_seq_length", default=512, type=int, required=True)
    args = parser.parse_args()

    file = args.dataset_dir+'{}_clean_new'.format(args.file_type)


    if args.dataset == 'pm20k' or args.dataset == 'pm200k':
        structure = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
    else:
        structure = ["background", "population", "intervention", "outcome", "study design", "other"]

    max_seq_length = 512

    output = dict()
    output['data'] = list()
    output['version'] = '1.0'
    total_abstract = 0
    abstract_in_max_seq_length = 0

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    with open(os.path.join(args.dataset_dir, '{}_clean_new.txt'.format(args.file_type)), 'r') as f, \
         open('{}_clean_new_final_{}AAAAAAA.json'.format(file, args.max_seq_length), 'w') as outfile:
        total_dict = dict()
        title = str()
        ids = 0
        for num, line in tqdm(enumerate(f), ascii=True):
            if line.startswith('###'):
                if num:
                    data = dict()
                    data['title'] = title
                    data['paragraphs'] = list()
                    paragraphs = dict()
                    paragraphs['context'] = ' '.join([total_dict[l] for l in structure if l in total_dict]).strip() # abstract
                    paragraphs['qas'] = list()
                    if len(tokenizer.tokenize(paragraphs['context'].strip())) < args.max_seq_length-4: # 4: Maximum condition -> [CLS] LABEL [SEP] ... [SEP]
                        for lab in structure:
                            if lab not in total_dict:
                                continue
                            qa = dict()
                            text = total_dict[lab].strip()
                            qa['answers'] = [{'answer_start': paragraphs['context'].index(text), 'text': text}]
                            qa['id'] = str(ids)
                            qa['question'] = lab
                            paragraphs['qas'].append(qa)
                            ids += 1
                        abstract_in_max_seq_length += 1
                        data['paragraphs'].append(paragraphs)
                        output['data'].append(data)
                total_abstract += 1
                title = line.strip()[3:]
                total_dict = dict()
            else:
                if line.strip():
                    instance = line.strip().split('\t')
                    print(instance)
                    label = instance[0]
                    sentence = instance[1]
                    if label not in total_dict:
                        total_dict[label] = sentence+' '
                    else:
                        total_dict[label] += sentence+' '
        json.dump(output, outfile, ensure_ascii=False, indent=4)
        print('Total Abstracrts:', total_abstract)
        print('max_seq_length(tokenization) < "{}": {}'.format(args.max_seq_length, abstract_in_max_seq_length))


if __name__ == '__main__':
    main()