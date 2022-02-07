# wav2vec2-turkish
Turkish Automated Speech Recognition (ASR) using Facebook's Wav2vec 2.0 models

## Fine-tuned Models
The following Wav2vec 2.0 models were finetuned during Huggingface's [Robust Speech Challenge](https://github.com/huggingface/transformers/tree/master/examples/research_projects/robust-speech-event) event: 
1. [mpoyraz/wav2vec2-xls-r-300m-cv6-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv6-turkish) achives 8.83 % WER on Common Voice 6.1 TR test split
2. [mpoyraz/wav2vec2-xls-r-300m-cv7-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish) achives 8.62 % WER on Common Voice 7 TR test split
3. [mpoyraz/wav2vec2-xls-r-300m-cv8-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv8-turkish) achives 10.61 % WER on Common Voice 8 TR test split

## Datasets
The following open source speech corpora is available for Turkish:
1. [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)
2. [MediaSpeech](https://www.openslr.org/108/)

This repo contains pre-processing and training scripts for these corpora.

## Pre-processing Datasets
After downloading Turkish speech corpora above, `preprocess.py` can be used to create datasets files for training.
- The script handles the text normalization required for proper training.
- Common Voice TR corpus is handled as follows:
   - Train split: all samples in `validated` split except `dev` and `test` samples is reserved to training.
   - Validation split: same as `dev` split.
   - Test split: same as `test` split.
- Media Speech corpus is fully included in the final train split if provided.
- Final datasets CSV files with 'path' & 'sentence' columns are saved to the output directory: `train.csv`, `validation.csv` and `test.csv`

```bash
python preprocess.py \
    --vocab vocab.json \
    --cv_path data/cv-corpus-<version>-<date>/tr \
    --media_speech_path data/TR \
    --output data \
```
## Training
[facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) is large-scale multilingual pretrained model for speech and used for fine-tuning on Turkish speech corpora. The exact hyperparameters used are available at model card on each finetuned model on Huggingface model hub.

An example training command:

```bash
python train_asr.py \
    --model_name_or_path facebook/wav2vec2-xls-r-300m \
    --vocab_path vocab.json \
    --train_file train_validation.csv \
    --validation_file test.csv \
    --output_dir exp \
    --audio_path_column_name path \
    --text_column_name sentence \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --eval_metrics wer cer \
    --freeze_feature_extractor \
    --mask_time_prob 0.1 \
    --mask_feature_prob 0.1 \
    --attention_dropout 0.05 \
    --activation_dropout 0.05 \
    --feat_proj_dropout 0.05 \
    --final_dropout 0.1 \
    --learning_rate 2.5e-4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --warmup_steps 500 \
    --eval_steps 500 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --save_total_limit 2 \
    --gradient_checkpointing \
    --fp16 \
    --group_by_length \
    --do_train \
    --do_eval \
```
## Evaluation
The following finetuned models are available on Huggingface model hub and has an evaluation script `eval.py` with appropiate text normalization. The commands for running evaluations are also available on the model cards.
1. [mpoyraz/wav2vec2-xls-r-300m-cv6-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv6-turkish) achives 8.83 % WER on Common Voice 6.1 TR test split
2. [mpoyraz/wav2vec2-xls-r-300m-cv7-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish) achives 8.62 % WER on Common Voice 7 TR test split
3. [mpoyraz/wav2vec2-xls-r-300m-cv8-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv8-turkish) achives 10.61 % WER on Common Voice 8 TR test split

## Language Model
For CTC beam search decoding with shallow LM fusion, n-gram language model is trained on a Turkish Wikipedia articles using KenLM and [ngram-lm-wiki](https://github.com/mpoyraz/ngram-lm-wiki) repo was used to generate arpa LM and convert it into binary format.
