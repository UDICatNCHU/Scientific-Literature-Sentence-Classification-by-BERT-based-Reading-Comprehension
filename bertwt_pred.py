import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
import collections
import logging
import os
import json
import re
import string
import time
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from collections import Counter
import itertools
import argparse


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 title_id,
                 question_tag_list,
                 doc_tokens,
                 answer_text_list=None):
        self.title_id = title_id
        self.question_tag_list = question_tag_list
        self.doc_tokens = doc_tokens
        self.answer_text_list = answer_text_list

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "title_id: %s" % (self.title_id)
        s += ", question_tag_list: %s" % (
            self.question_tag_list)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += ", answer_text_list: %s" % (
            self.answer_text_list)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 output_ids,
                 example_index):
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.output_ids = output_ids
        self.example_index = example_index


def read_squad_examples(input_file, dataset_name):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []

    if dataset_name == 'pm20k' or dataset_name == 'pm200k':
        convert_table = {"BACKGROUND": "[B]", 
                         "OBJECTIVE": "[O]", 
                         "METHODS": "[M]", 
                         "RESULTS": "[R]", 
                         "CONCLUSIONS": "[C]"}
    else:
        convert_table = {"background": "[background]", 
                         "intervention": "[intervention]", 
                         "other": "[other]", 
                         "outcome": "[outcome]", 
                         "population": "[population]", 
                         "study design": "[study design]"}
        
    for entry in input_data:
        title_id = entry['title']
        question_tag_list = []
        answer_text_list = []
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_tag = convert_table[qa["question"]]


                answer = qa["answers"][0]
                answer_text = answer["text"]
                question_tag_list.append(question_tag)
                answer_text_list.append(answer_text)

            example = SquadExample(
                title_id=title_id,
                question_tag_list=question_tag_list,
                doc_tokens=doc_tokens,
                answer_text_list=answer_text_list)
            examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, show_all_examples=False):
    """Loads a data file into a list of `InputBatch`s."""
    
    features = []
    for (example_index, example) in enumerate(examples):

        all_doc_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # The -2 accounts for [CLS], [SEP]
        max_tokens_for_doc = max_seq_length - 2

        if max_tokens_for_doc <= len(all_doc_tokens): continue

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            output_tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            # Output
            output_tokens.append("[CLS]")
            for i in range(len(example.question_tag_list)):
                question_tag = example.question_tag_list[i]
                answer_tokens = tokenizer.tokenize(example.answer_text_list[i])
                for _ in answer_tokens:
                    output_tokens.append(question_tag)
            output_tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            # (-1)-pad up to the sequence length.
            while len(output_ids) < max_seq_length:
                output_ids.append(-1)
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(output_ids) == max_seq_length

            if example_index < 20 or show_all_examples:
                logger.info("*** Example ***")
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))

            features.append(
                InputFeatures(
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    output_ids=output_ids,
                    example_index=example_index
                    ))
    return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    score = metric_fn(prediction, ground_truths)
    return score


class LM_Pred(object):
    """docstring for LM_Pred"""
    def __init__(self, bert_model, dataset, dataset_dir, predict_file, output_dir):
        self.max_seq_length = 512
        self.doc_stride = 512
        self.show_all_examples = False
        self.start_epoch = 1
        self.end_epoch = 50

        self.bert_model = bert_model
        self.dataset = dataset # pm20k, pm200k, nicta
        self.dataset_dir = dataset_dir
        self.predict_file = predict_file
        self.output_dir = output_dir

        os.environ["CUDA_VISIBLE_DEVICES"] = "1"


        if self.dataset == 'pm20k':
            self.labels = ['[B]', '[O]', '[M]', '[R]', '[C]']
        elif self.dataset == 'pm200k':
            self.labels = ['[B]', '[O]', '[M]', '[R]', '[C]']
        else:
            self.labels = ['[background]', '[population]', '[intervention]', '[outcome]', '[study design]', '[other]']

        self.input_file = os.path.join(self.dataset_dir, self.predict_file)


        ans_filename = 'test_clean_new_ans.json'
        ans_path = os.path.join(self.dataset_dir, ans_filename)
        if not os.path.exists(ans_path):
            self.create_ans(ans_filename)
        self.ans_json = self.loadFile(ans_path)

        self.result_map = dict()


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
                elif not line.strip():
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
        if self.dataset == 'pm20k' or self.dataset == 'pm200k':
            labels = ['[B]', '[O]', '[M]', '[R]', '[C]']
        else:
            labels = ["[background]", "[population]", "[intervention]", "[outcome]", "[study design]", "[other]"]

        if len(counter) == 0:
            pred_label = []
        elif len(counter) % 2 == 0:
            if counter.most_common()[0][1] != counter.most_common()[1][1]:
                pred_label = [counter.most_common(1)[0][0]]
            else:
                try:
                    pred_label = [labels[min(labels.index(counter.most_common()[0][0]), labels.index(counter.most_common()[1][0]))]]
                except:
                    pred_label = ['[C]']
        else:
            if len(counter) == 1 or counter.most_common()[0][1] != counter.most_common()[1][1]:
                pred_label = [counter.most_common(1)[0][0]]
            else:
                if len(set([counter[ele] for ele in counter])) == 1:
                    pred_label = [labels[min([labels.index(ele) for ele in counter])]]
                else:
                    try:
                        pred_label = [labels[min(labels.index(counter.most_common()[0][0]), labels.index(counter.most_common()[1][0]))]]
                    except:
                        pred_label = ['[C]']
        return pred_label


    def get_sentence_based_rsult_map(self, predictions):

        tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        print('Sentence-Based.')

        result_map = dict()
        for abstract_title in predictions[0]: 
            if abstract_title not in result_map:
                result_map[abstract_title] = list()

            start = end = 0
            for ele in self.ans_json[abstract_title]:
                sentence_tokens = tokenizer.tokenize(ele['text'])
                start = end
                end = start+len(sentence_tokens)
                
                pred_label_list = []
                for p in predictions:
                    p_sentence = p[abstract_title]['result_map'][start:end]
                    p_counter = Counter(list(itertools.chain.from_iterable(p_sentence)))
                    pred_label = self.get_pred_label(p_counter)
                    pred_label_list += pred_label

                counter = Counter(pred_label_list)
                result_map[abstract_title].append(self.get_pred_label(counter))
            assert len(result_map[abstract_title]) == len(self.ans_json[abstract_title])
        
        # json.dump([ele for ele in new_result_map], open('./title_list.json', 'w'), ensure_ascii=False, indent=4)

        return result_map


    def evaluate(self, ground_truths, predictions):

        if self.dataset == 'pm20k' or self.dataset == 'pm200k':
            convert_table = {"BACKGROUND": "[B]", 
                             "OBJECTIVE": "[O]", 
                             "METHODS": "[M]", 
                             "RESULTS": "[R]", 
                             "CONCLUSIONS": "[C]"}
        else:
            convert_table = {"background": "[background]", 
                             "population": "[population]", 
                             "intervention": "[intervention]", 
                             "outcome": "[outcome]", 
                             "study design": "[study design]",
                             "other": "[other]"}

        all_y_true = []
        all_y_pred = []
        for abstract_title in predictions:
            ground_truth_list = [convert_table[ele['label']] for ele in ground_truths[abstract_title]]
            prediction_list = list(itertools.chain.from_iterable(predictions[abstract_title]))

            assert len(ground_truth_list) == len(prediction_list)

            all_y_true += ground_truth_list
            all_y_pred += prediction_list

        # json.dump(all_y_true, open('./y_true.json', 'w'), ensure_ascii=False, indent=4)
        # json.dump(all_y_pred, open('./y_pred.json', 'w'), ensure_ascii=False, indent=4)

        _, _, weighted_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='weighted')
        _, _, micro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='micro')
        _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro')

        class_report = classification_report(all_y_true, all_y_pred, digits=4, target_names=self.labels, labels=self.labels)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=self.labels)

        print(class_report)
        print()
        print(cm)
        print()
        print('weighted_f1:', weighted_f1)
        print('micro_f1:', micro_f1)
        print('macro_f1:', macro_f1)
        print()


    def get_pred_file(self):
        device = torch.device("cuda")

        tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        examples = read_squad_examples(
                input_file=self.input_file, dataset_name=self.dataset)
        features = convert_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=self.max_seq_length,
                    doc_stride=self.doc_stride, 
                    show_all_examples=self.show_all_examples)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(examples))
        logger.info("  Num split examples = %d", len(features))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_output_ids = torch.tensor([f.output_ids for f in features], dtype=torch.long)
        all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_output_ids, all_example_indices)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

        result = []
        trained_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
        model_state_dict = torch.load(trained_model_path)
        model = BertForMaskedLM.from_pretrained(self.bert_model, state_dict=model_state_dict)

        model.to(device)
        model.eval()

        sentence_results = dict()
        pred_result = dict()

        all_y_true = []
        all_y_pred = []

        for input_ids, input_mask, segment_ids, output_ids, example_index in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            predictions = model(input_ids, segment_ids, input_mask)
            
            title_id = examples[example_index].title_id
            input_text_list = []
            predictions_tag_list = []
            output_tag_list = []
            sentence_results[title_id] = list()
            for i in range(len(input_ids[0])):
                predicted_index = torch.argmax(predictions[0][i]).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
                input_token = tokenizer.convert_ids_to_tokens([input_ids[0][i].item()])
                output_token = tokenizer.convert_ids_to_tokens([output_ids[0][i].item()])
                if input_token[0] == '[CLS]': continue
                if input_token[0] == '[SEP]': break
                input_text_list.append(input_token[0])
                predictions_tag_list.append(predicted_token[0])
                output_tag_list.append(output_token[0])

            if title_id not in pred_result:
                pred_result[title_id] = dict()
                pred_result[title_id]['result_map'] = [[ele] for ele in predictions_tag_list]
                pred_result[title_id]['sentence_truths'] = list()
                pred_result[title_id]['sentence_preds'] = list()

                result_text = list()
                temp = list()
                for i in range(len(predictions_tag_list)):
                    if i > 0 and predictions_tag_list[i] != predictions_tag_list[i-1]:
                        result_text.append({predictions_tag_list[i-1]: " ".join(temp)})
                        temp = list()
                        temp.append(input_text_list[i])
                    else:
                        temp.append(input_text_list[i])
                result_text.append({predictions_tag_list[-1]: " ".join(temp)})
                pred_result[title_id]['result_text'] = result_text
            
            start = end = 0
            for ele in self.ans_json[title_id]:
                text = tokenizer.tokenize(ele['text'])
                start = end
                end = start+len(text)
                g_counter = Counter(output_tag_list[start:end])
                p_counter = Counter(predictions_tag_list[start:end])

                y_true = self.get_pred_label(g_counter)
                y_pred = self.get_pred_label(p_counter)
                all_y_true += y_true
                all_y_pred += y_pred

                pred_result[title_id]['sentence_truths'].append(y_true)
                pred_result[title_id]['sentence_preds'].append(y_pred)

        _, _, weighted_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='weighted')
        _, _, micro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='micro')
        _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro')
        tmp = {'weighted_f1': weighted_f1, 
               'micro_f1': micro_f1, 
               'macro_f1': macro_f1}
        result.append(tmp)


        class_report = classification_report(all_y_true, all_y_pred, digits=4, target_names=self.labels, labels=self.labels)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=self.labels)
        
        metrics = {'classification-report': class_report, 'confusion-matrix': cm}

        json.dump(pred_result, open(os.path.join(self.output_dir, 'predictions_test.json'), 'w'), ensure_ascii=False, indent=4)
        output_model_result_file = os.path.join(self.output_dir, "model_result.txt")
        json.dump(result, open(output_model_result_file, "w"), ensure_ascii=False, indent=4)
        with open(os.path.join(self.output_dir, 'test_reports.txt'), 'a') as file:
            file.write('=============== classification-report ===============\n')
            file.write('{}\n\n\n'.format(metrics['classification-report']))
            file.write('================= confusion-matrix =================\n')
            file.write('{}\n'.format(metrics['confusion-matrix']))
            file.write('\nweighted_f1: {}\nmicro_f1: {}\nmacro_f1: {}\n'.format(tmp['weighted_f1'], tmp['micro_f1'], tmp['macro_f1']))
            file.write('----------------------------------------------------\n\n\n')

        print('\n=======================================================')
        print(tmp)
        print('=======================================================')
        print(metrics['classification-report'])
        print('=======================================================')
        print(metrics['confusion-matrix'])


    def get_eval_result(self):
        predictions = [self.loadFile(os.path.join(self.output_dir, 'predictions_test.json'))]
        self.result_map = self.get_sentence_based_rsult_map(predictions=predictions)
        self.evaluate(ground_truths=self.ans_json, predictions=self.result_map)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bert_model", default=None, type=str, required=True)
    parser.add_argument("--dataset", default='pm20k', type=str, required=True, help="pm20k, pm200k, or nicta.")
    parser.add_argument("--dataset_dir", default='data/PubMed_20k_RCT/', type=str, required=True, help="Path to the dataset dir.")
    parser.add_argument("--predict_file", default='test_clean_new_final_512.json', type=str, required=True, help="Path to the predict file.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="Path to the output dir.")

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--end_epoch", default=0, type=int)
    
    args = parser.parse_args()

    obj = LM_Pred(bert_model=args.bert_model, dataset=args.dataset, dataset_dir=args.dataset_dir, 
                  predict_file=args.predict_file, output_dir=args.output_dir)
    obj.get_pred_file()
    # obj.get_eval_result()



if __name__ == '__main__':
    main()