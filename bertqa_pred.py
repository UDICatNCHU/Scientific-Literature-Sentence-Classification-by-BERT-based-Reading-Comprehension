import json
import os
import time
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import itertools
import argparse

class SQuAD_Pred(object):
    """docstring for SQuAD_Pred"""
    def __init__(self, dataset, dataset_dir):
        self.dataset = dataset # pm20k, pm200k, nicta
        self.dataset_dir = dataset_dir

        if self.dataset == 'pm20k':
            self.labels = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
        elif self.dataset == 'pm200k':
            self.labels = ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
        else:
            self.labels = ["background", "population", "intervention", "outcome", "study design", "other"]

        self.result_map = dict()

        # 讀取正確答案, 若檔案沒存在則建立
        ans_filename = 'test_clean_new_ans.json'
        ans_path = os.path.join(self.dataset_dir, ans_filename)
        if not os.path.exists(ans_path):
            self.create_ans(ans_filename)
        self.ans_json = self.loadFile(ans_path)


    def loadFile(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data


    def create_ans(self, ans_filename):
        sentence_id = 0
        output = dict()
        with open(os.path.join(self.dataset_dir, 'test_clean_new.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('###'):
                    abstract_title = line.strip().split('#')[-1]
                    if abstract_title not in output:
                        output[abstract_title] = list()
                        temp = list()
                elif not line.strip(): # 整理順序並寫入output
                    res = {}
                    for item in temp:
                        res.setdefault(item['label'], []).append(item)
                    for label in self.labels:
                        if label in res:
                            for ele in res[label]:
                                output[abstract_title].append(ele)
                else:
                    label, sentence = line.strip().split('\t')
                    example = {'sentence_id': sentence_id, 
                               'label': label, 
                               'text': sentence}
                    temp.append(example)
                    sentence_id += 1
                    assert label in self.labels
                    assert len(line.strip().split('\t')) == 2
        json.dump(output, open(os.path.join(self.dataset_dir, ans_filename), 'w'), ensure_ascii=False, indent=4)


    def get_pred_label(self, counter):
        if len(counter) == 0:
            pred_label = []
        elif len(counter) % 2 == 0:
            if counter.most_common()[0][1] != counter.most_common()[1][1]:
                pred_label = [counter.most_common(1)[0][0]]
            else:
                pred_label = [self.labels[min(self.labels.index(counter.most_common()[0][0]), self.labels.index(counter.most_common()[1][0]))]]
        else: 
            if len(counter) == 1 or counter.most_common()[0][1] != counter.most_common()[1][1]:
                pred_label = [counter.most_common(1)[0][0]]
            else: 
                if len(set([counter[ele] for ele in counter])) == 1:
                    pred_label = [self.labels[min([self.labels.index(ele) for ele in counter])]]
                else:
                    pred_label = [self.labels[min(self.labels.index(counter.most_common()[0][0]), self.labels.index(counter.most_common()[1][0]))]]
        return pred_label


    def get_sentence_based_rsult_map(self, predictions):
        print('Sentence-Based.')
        result_map = dict()
        for abstract_title in predictions[0]:
            if abstract_title not in result_map:
                result_map[abstract_title] = list()
            start = end = 0
            for ele in self.ans_json[abstract_title]:
                sentence_tokens = ele['text'].split()
                start = end
                end = start+len(sentence_tokens)
                
                pred_label_list = []
                for p in predictions:
                    p_sentence = p[abstract_title]['result_map'][start:end]
                    p_counter = Counter()
                    for tags in p_sentence:
                        p_counter += Counter(tags)
                    pred_label = self.get_pred_label(p_counter)
                    pred_label_list += pred_label

                counter = Counter(pred_label_list)
                result_map[abstract_title].append(self.get_pred_label(counter))
            assert len(result_map[abstract_title]) == len(self.ans_json[abstract_title])

        new_result_map = dict()
        error_count = 0
        error_title_list = []
        error_label_list = []
        for ele in result_map:
            assert len(self.ans_json[ele]) == len(result_map[ele])
            for i, e in enumerate(result_map[ele]):
                if len(e) == 0:
                    if i == 0:
                        index_in_labels = self.labels.index(list(itertools.chain.from_iterable(result_map[ele]))[0])
                        if index_in_labels != 0:
                            result_map[ele][i] = [self.labels[index_in_labels-1]]
                        else:
                            result_map[ele][i] = [self.labels[0]]
                    else:    
                        result_map[ele][i] = result_map[ele][i-1]
        
        # json.dump([ele for ele in result_map], open('./abstract_title_list.json', 'w'), ensure_ascii=False, indent=4)

        return result_map


    def evaluate(self, ground_truths, predictions):
        all_y_true = []
        all_y_pred = []

        for abstract_title in predictions:
            ground_truth_list = [ele['label'] for ele in ground_truths[abstract_title]]
            prediction_list = list(itertools.chain.from_iterable(predictions[abstract_title]))

            assert len(ground_truth_list) == len(prediction_list)

            all_y_true += ground_truth_list
            all_y_pred += prediction_list

        _, _, weighted_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='weighted')
        _, _, micro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='micro')
        _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro')

        if self.dataset == 'pm20k' or self.dataset == 'pm200k':
            class_report = classification_report(all_y_true, all_y_pred, digits=4, target_names=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'], labels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'])
            cm = confusion_matrix(all_y_true, all_y_pred, labels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'])
        else:
            class_report = classification_report(all_y_true, all_y_pred, digits=4, target_names=["background", "population", "intervention", "outcome", "study design", "other"], labels=["background", "population", "intervention", "outcome", "study design", "other"])
            cm = confusion_matrix(all_y_true, all_y_pred, labels=["background", "population", "intervention", "outcome", "study design", "other"])

        print(class_report)
        print()
        print(cm)
        print()
        print('weighted_f1:', weighted_f1)
        print('micro_f1:', micro_f1)
        print('macro_f1:', macro_f1)
        print()

        return class_report, cm, weighted_f1, micro_f1, macro_f1


    def get_eval_score(self, prediction_file):
        predictions = [self.loadFile(prediction_file)]
        self.result_map = self.get_sentence_based_rsult_map(predictions=predictions)
        self.evaluate(ground_truths=self.ans_json, predictions=self.result_map)



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", default='pm20k', type=str, required=True, help="pm20k, pm200k, or nicta.")
    parser.add_argument("--dataset_dir", default='data/PubMed_20k_RCT/', type=str, required=True, help="Path to the dataset dir.")
    parser.add_argument("--prediction_file", default='predictions_test.json', type=str, required=True, help="Path to the prediction file.")

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=0, type=int)
    
    args = parser.parse_args()

    obj = SQuAD_Pred(dataset=args.dataset, dataset_dir=args.dataset_dir)
    obj.get_eval_score(prediction_file=args.prediction_file)


if __name__ == '__main__':
    main()