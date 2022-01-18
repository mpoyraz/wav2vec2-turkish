import re
import sys
import logging
import argparse
import torch
import torchaudio
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
)
from datasets import DatasetDict, load_dataset, load_metric, set_caching_enabled
from utils import remove_special_characters, unify_characters

set_caching_enabled(False)
logger = logging.getLogger(__name__)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate Wav2vec ASR with CTC")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, default="common_voice",
                        help="The configuration name of the dataset to use (via the datasets library)")
    parser.add_argument("--dataset_config_name", type=str, default="tr",
                        help="The configuration name of the dataset to use (via the datasets library)")
    parser.add_argument("--eval_split_name", type=str, default="test",
                        help="The name of the evaluation data set split to use (via the datasets library)")
    parser.add_argument("--audio_column_name", type=str, default="audio",
                        help="The name of the dataset column containing the audio data")
    parser.add_argument("--text_column_name", type=str, default="sentence",
                        help="The name of the dataset column containing the text data")
    parser.add_argument("--preprocessing_num_workers", type=int, default=1,
                        help="The number of processes to use for the preprocessing")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="The batch size for evaluation")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    
    # Torch device to run evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using {} for evaluation".format(device))
    
    # Wav2vec2 processor and model
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForCTC.from_pretrained(args.model_name_or_path)
    model = model.to(device)
    
    # Load evaluation dataset
    eval_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.eval_split_name)
    logging.info("Dataset '{}' - split '{}' has {} records".format(
        args.dataset_name, args.eval_split_name, len(eval_dataset)))
    
    # Preprocess text and resample audio
    def preprocess(sample):
        # Lowercase and clean text
        text = sample[args.text_column_name].lower()
        text = remove_special_characters(text)
        text = unify_characters(text)
        text = re.sub('\s+', ' ', text)
        sample['text'] = text
        # Resample audio array
        resampler = torchaudio.transforms.Resample(
            sample[args.audio_column_name]["sampling_rate"],
            processor.feature_extractor.sampling_rate
        )
        array_pt = torch.from_numpy(sample[args.audio_column_name]["array"]).unsqueeze(0)
        sample['audio_array'] = resampler(array_pt).squeeze().numpy()
        return sample
        
    eval_dataset = eval_dataset.map(
        preprocess, num_proc=args.preprocessing_num_workers
    )

    # Predict on eval dataset
    def predict(batch):
        inputs = processor(batch['audio_array'],
                           sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="pt", padding=True)
        # Move torch tensor to the device
        for k in inputs.keys():
            if inputs[k] is not None and torch.is_tensor(inputs[k]):
                inputs[k] = inputs[k].to(device)
        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits
            # No LM case
            # pred_ids = torch.argmax(logits, dim=-1)
            # batch["pred_strings"] = processor.batch_decode(pred_ids)
            # With LM
            decode_results = processor.batch_decode(logits.cpu().numpy())
            batch["pred_strings"] = decode_results.text
            
        return batch
    
    eval_dataset = eval_dataset.map(
        predict, batched=True, batch_size=args.batch_size
    )
    
    # Load metrics and calculate on eval dataset
    wer, cer = load_metric("wer"), load_metric("cer")
    wer_score = wer.compute(predictions=eval_dataset["pred_strings"], references=eval_dataset["text"])
    cer_score = cer.compute(predictions=eval_dataset["pred_strings"], references=eval_dataset["text"])
    logging.info("WER: {:.2f} % , CER: {:.2f} %".format(100*wer_score, 100*cer_score))    
    
if __name__ == "__main__":
    main()