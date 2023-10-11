# svp-swap
Code for "Swap-and-Predict -- Unsupervised Lexical Semantic Change Detection across Corpora by Context Swapping", Findings of EMNLP2023

- Data: 
  - [SemEval-2020 Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/)
  - [Liverpool FC](https://github.com/marcodel13/Short-term-meaning-shift)
- Pretrained MLM: [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
- Finetuned MLM: [TempoBERT](https://github.com/guyrosin/tempobert)
- Source Code:
  - `add_time_tokens.py`
  - `convert_liverpoolfc_to_semeval.py`
  - `main.py`
  - `requirements.txt`
  - `utils.py`
  - `wordvec_from_bert.py`


## 1\. Setup
### From requirements.txt
Python `>= 3.8`
```
pip install -r requirements.txt
```

### From Dockerfile
```
docker build -t svp-swap .

docker run -it svp-swap
```

## 2\. Obtain Sibling Embeddings from MLM
- last 4 layers: `hidden_layers == 4`

For liverpool fc, preprocessing is required
```
python3 convert_liverpoolfc_to_semeval.py \
    --word_path {PATH_TO_TARGET_FILE} \
    --file_pathes {PATH_TO_CORPUS_C1} {PATH_TO_CORPUS_C2}
```

### Pretrained MLM (multilingual BERT, cased)
```
python3 wordvec_from_bert.py \
    --file_path {PATH_TO_CORPUS_C1} \
    --target_word_list {PATH_TO_TARGET_WORD_LIST} \
    --hidden_layers 4 \
    --output_name pretrained_4layers_c1 

python3 wordvec_from_bert.py \
    --file_path {PATH_TO_CORPUS_C2} \
    --target_word_list {PATH_TO_TARGET_WORD_LIST} \
    --hidden_layers 4 \
    --output_name pretrained_4layers_c2 
```

### Finetuned MLM
add time tokens <1> <2> to target corpora
```
python3 add_time_tokens.py \
    --file_pathes {PATH_TO_CORPUS_C1} {PATH_TO_CORPUS_C2}
```

```
python3 wordvec_from_bert.py \
    --file_path {PATH_TO_CORPUS_C1_ADDED_TIME_TOKENS} \
    --target_word_list {PATH_TO_TARGET_WORD_LIST} \
    --hidden_layers 4 \
    --is_finetuned \
    --finetuned_tokenizer {PATH_TO_TOKENIZER.JSON} \
    --finetuned_vocab {PATH_TO_VOCAB.TXT} \
    --finetuned_config {PATH_TO_CONFIG.JSON} \
    --finetuned_model {PATH_TO_MODEL.BIN} \
    --output_name finetuned_4layers_c1 

python3 wordvec_from_bert.py \
    --file_path {PATH_TO_CORPUS_C2_ADDED_TIME_TOKENS} \
    --target_word_list {PATH_TO_TARGET_WORD_LIST} \
    --hidden_layers 4 \
    --is_finetuned \
    --finetuned_tokenizer {PATH_TO_TOKENIZER.JSON} \
    --finetuned_vocab {PATH_TO_VOCAB.TXT} \
    --finetuned_config {PATH_TO_CONFIG.JSON} \
    --finetuned_model {PATH_TO_MODEL.BIN} \
    --output_name finetuned_4layers_c2 
```

## 3\. Make Prediction
 - random swapping: `shuffle_func=random`
 - distance-based swapping: `shuffle_func=distance`
 - consider the ratio of corpus size: `--pathes_corpora {PATH_TO_CORPUS_C1} {PATH_TO_CORPUS_C2}`

### Pretrained MLM
```
python3 main.py \
    --wordvec_pathes results/pretrained_4layers_c1 \
    results/pretrained_4layers_c2 \
    --graded_words_list {PATH_TO_GOLD} \
    --output_name pretrained_4layers
```

### Finetuned MLM
```
python3 main.py \
    --wordvec_pathes results/finetuned_4layers_c1 \
    results/finetuned_4layers_c2 \
    --graded_words_list {PATH_TO_GOLD} \
    --output_name pretrained_4layers
```
