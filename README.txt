1. Please download the complete code and datasets from the link below and place it in accordance with the "dir_tree.txt" rule. 
   In addition to the "results" model can be selectively downloaded on demand, the rest is required.
>>> https://tinyurl.com/y6s775nt

2. Install the required package.
>>> pip install -r requirements.txt

3. Start testing/training the model.
=========================================
Naive-BERT:

TESTING
(1) PubMed 20k
python3 naive_and_bertc.py \
  --task_name pm20k \
  --do_eval \
  --data_dir data/PubMed_20k_RCT/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/Naive-BERT_pm20k/

(2) PubMed 200k
python3 naive_and_bertc.py \
  --task_name pm200k \
  --do_eval \
  --data_dir data/PubMed_200k_RCT/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/Naive-BERT_pm200k/

(3) NICTA
python3 naive_and_bertc.py \
  --task_name nicta \
  --do_eval \
  --data_dir data/nicta_piboso/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/Naive-BERT_nicta/

-----------------------------------------

TRAINING
(1) PubMed 20k
python3 naive_and_bertc.py \
  --task_name pm20k \
  --do_train \
  --data_dir data/PubMed_20k_RCT/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

(2) PubMed 200k
python3 naive_and_bertc.py \
  --task_name pm200k \
  --do_train \
  --data_dir data/PubMed_200k_RCT/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

(3) nicta
python3 naive_and_bertc.py \
  --task_name nicta \
  --do_train \
  --data_dir data/nicta_piboso/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

=========================================

BERT-C:

TESTING
(1) PubMed 20k
python3 naive_and_bertc.py \
  --task_name seq_pm20k \
  --do_eval \
  --do_lower_case \
  --data_dir data/PubMed_20k_RCT/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/BERT-C_pm20k/

(2) PubMed 200k
python3 naive_and_bertc.py \
  --task_name seq_pm200k \
  --do_eval \
  --do_lower_case \
  --data_dir data/PubMed_200k_RCT/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/BERT-C_pm200k/

(3) NICTA
python3 naive_and_bertc.py \
  --task_name seq_nicta \
  --do_eval \
  --do_lower_case \
  --data_dir data/nicta_piboso/ \
  --bert_model bert-base-uncased/ \
  --output_dir results/BERT-C_nicta/

-----------------------------------------

TRAINING
(1) PubMed 20k
python3  naive_and_bertc.py \
  --task_name seq_pm20k \
  --do_train \
  --data_dir data/PubMed_20k_RCT/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

(2) PubMed 200k
python3  naive_and_bertc.py \
  --task_name seq_pm200k \
  --do_train \
  --data_dir data/PubMed_200k_RCT/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

(3) nicta
python3  naive_and_bertc.py \
  --task_name seq_nicta \
  --do_train \
  --data_dir data/nicta_piboso/ \
  --bert_model bert-base-uncased \
  --max_seq_length 512 \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir "path/to/output_dir"

=========================================
BERT-WT:

TESTING
(1) pm20k
python3 bertwt_pred.py \
  --bert_model bert-base-uncased/ \
  --dataset pm20k \
  --dataset_dir data/PubMed_20k_RCT/ \
  --predict_file test_clean_new_final_512.json \
  --output_dir "path/to/output_dir"

(2) pm200k
python3 bertwt_pred.py \
  --bert_model bert-base-uncased/ \
  --dataset pm200k \
  --dataset_dir data/PubMed_200k_RCT/ \
  --predict_file test_clean_new_final_512.json \
  --output_dir "path/to/output_dir"

(3) nicta
python3 bertwt_pred.py \
  --bert_model bert-base-uncased/ \
  --dataset pm200k \
  --dataset_dir data/nicta_piboso/ \
  --predict_file test_clean_new_final_512.json \
  --output_dir "path/to/output_dir"

-----------------------------------------

TRAINING
(1) pm20k
python3 bertwt_train.py \
  --task pm20k \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/PubMed_20k_RCT/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 80 \
  --max_seq_length 512 \
  --train_batch_size 10 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"

(2) pm200k
python3 bertwt_train.py \
  --task pm200k \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/PubMed_200k_RCT/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 80 \
  --max_seq_length 512 \
  --train_batch_size 10 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"

(3) nicta
python3 bertwt_train.py \
  --task nicta \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/nicta_piboso/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 80 \
  --max_seq_length 512 \
  --train_batch_size 10 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"


=========================================
BERT-QA:

TESTING -> 2 step (a) Create prediction file. (b) Load prediction file to calculate score.
(1) pm20k
(a)
  python3 bertqa.py \
    --bert_model bert-base-uncased/ \
    --do_predict \
    --do_lower_case \
    --predict_file data/PubMed_20k_RCT/test_clean_new_final_512.json \
    --output_dir results/BERT-QA_pm20k/

(b)
  python3 bertqa_pred.py \
    --dataset pm20k \
    --dataset_dir data/PubMed_20k_RCT/ \
    --prediction_file "path/to/predictions_test.json/created/by/(a)"

(2) pm200k
(a)
  python3 bertqa.py \
    --bert_model bert-base-uncased/ \
    --do_predict \
    --do_lower_case \
    --predict_file data/PubMed_200k_RCT/test_clean_new_final_512.json \
    --output_dir results/BERT-QA_pm200k/

(b)
  python3 bertqa_pred.py \
    --dataset pm200k \
    --dataset_dir data/PubMed_200k_RCT/ \
    --prediction_file "path/to/predictions_test.json/created/by/(a)"

(3) nicta
(a)
  python3 bertqa.py \
    --bert_model bert-base-uncased/ \
    --do_predict \
    --do_lower_case \
    --predict_file data/PubMed_200k_RCT/test_clean_new_final_512.json \
    --output_dir results/BERT-QA_pm200k/

(b)
  python3 bertqa_pred.py \
    --dataset nitca \
    --dataset_dir data/nicta_piboso/ \
    --prediction_file "path/to/predictions_test.json/created/by/(a)"

-----------------------------------------

TRAINING
(1) pm20k
python3 bertqa.py \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/PubMed_20k_RCT/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 15 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --train_batch_size 12 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"

(2) pm200k
python3 bertqa.py \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/PubMed_200k_RCT/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 15 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --train_batch_size 12 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"

(3) nicta
python3 bertqa.py \
  --bert_model bert-base-uncased/ \
  --do_train \
  --do_lower_case \
  --train_file data/nicta_piboso/train_clean_new_512.json \
  --learning_rate 3e-5 \
  --num_train_epochs 15 \
  --max_seq_length 512 \
  --doc_stride 256 \
  --train_batch_size 12 \
  --gradient_accumulation_steps 2 \
  --output_dir "path/to/output_dir"

=========================================

4. others

Get train/test tsv file. -> "train_clean_seq.tsv" and "test_clean_new_final_512.tsv"
python3 get_tsv.py \
  --dataset pm20k \
  --dataset_dir data/PubMed_20k_RCT/ \
  --file_type test

-----------------------------------------

Get "train_clean_new_512.json"
python3 get_json.py \
  --bert_model bert-base-uncased/ \
  --dataset pm20k \
  --dataset_dir data/PubMed_20k_RCT/ \
  --file_type train \
  --max_seq_length 512

----------------------------------------