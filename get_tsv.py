import json
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default='pm20k', type=str, required=True, help="pm20k, pm200k, or nicta.")
    parser.add_argument("--dataset_dir", default='data/PubMed_20k_RCT/', type=str, required=True, help="Path to the dataset dir.")
    parser.add_argument("--file_type", default='test', type=str, required=True, help='train, test, or dev.')
    
    args = parser.parse_args()

    file = args.dataset_dir+'{}_clean_new'.format(args.file_type)
    
    label_list = list()
    output = 'LABEL\tSENTENCE\tABSTRACT\n'


    with open('./abstract_title_list_{}.json'.format(args.dataset), 'r') as f:
        abstract_title_list = json.load(f)


    if args.file_type == 'test' or args.file_type == 'dev':
        output_file_name = '{}_clean_new_final_512.tsv'.format(args.file_type)
    else:
        output_file_name = '{}_clean_new_seq.tsv'.format(args.file_type)


    count = 0
    with open(os.path.join(args.dataset_dir, '{}_clean_new.txt'.format(args.file_type)), 'r') as f, \
         open(os.path.join(args.dataset_dir, output_file_name), 'w') as outfile:
        label_list = list()
        sentence_list = list()
        abstract = str()
        for num, line in enumerate(f):
            if line.startswith('###'):
                if num:
                    if args.file_type == 'test':
                        if abstract_title in abstract_title_list:
                            count += len(label_list)
                            for i in range(len(label_list)):
                                data = '{}\t{}\t{}\n'.format(label_list[i], sentence_list[i], abstract)
                                output += data
                    else:
                        for i in range(len(label_list)):
                            count += len(label_list)
                            data = '{}\t{}\t{}\n'.format(label_list[i], sentence_list[i], abstract)
                            output += data
                abstract_title = line.strip()[3:]
                abstract = str()
                label_list = list()
                sentence_list = list()
            else:
                if line.strip():
                    instance = line.strip().split('\t')
                    label = instance[0]
                    sentence = instance[1]
                    label_list.append(label)
                    sentence_list.append(sentence)
                    abstract += sentence
        outfile.write(output)
    print(count)

if __name__ == '__main__':
    main()
