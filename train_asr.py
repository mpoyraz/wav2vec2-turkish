import functools
import json
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
import datasets
from datasets import DatasetDict, load_dataset, load_metric, set_caching_enabled

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
set_caching_enabled(False)

def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    vocab_path: str = field(
        metadata={"help": "Path to ASR vocabulary, tokens as JSON file"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    attention_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."}
    )
    activation_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "The dropout ratio for activations inside the fully connected layer."}
    )
    feat_proj_dropout: Optional[float] = field(
        default=0.0, metadata={"help": "The dropout ratio for the projected features."}
    )
    hidden_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
            "vectors will be masked along the time axis."
        },
    )
    mask_time_length: Optional[int] = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
            "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: Optional[int] = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: Optional[float] = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean", metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: str = field(
        metadata={"help": "The training data file (CSV file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional evaluation data file to evaluate on (CSV file)."},
    )
    delimiter: Optional[str] = field(
        default=",",
        metadata={"help": "Specifies the character delimiting individual cells in the CSV data"},
    )
    audio_path_column_name: Optional[str] = field(
        default="path",
        metadata={"help": "The name of the dataset column containing the audio paths. Defaults to 'path'"},
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    eval_metrics: Optional[List[str]] = list_field(
        default=["wer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: Optional[float] = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    unk_token: Optional[str] = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: Optional[str] = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: Optional[str] = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load the model config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = training_args.output_dir

    with training_args.main_process_first():
        # Load vocab from file
        with open(model_args.vocab_path) as fp:
            vocab_dict = json.load(fp)

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")
        if training_args.overwrite_output_dir and os.path.isfile(vocab_file):
            os.remove(vocab_file)
        
        # Save vocab dict to be loaded into tokenizer
        if not os.path.isfile(vocab_file):
            with open(vocab_file, "w") as file:
                json.dump(vocab_dict, file)

    # Tokenizer args
    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
        "unk_token": data_args.unk_token,
        "pad_token": data_args.pad_token,
        "word_delimiter_token": data_args.word_delimiter_token,
    }

    # Load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs,
    )

    # Load feature_extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir,
    )
    sampling_rate_target = feature_extractor.sampling_rate

    # Update config for finetuning
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "bos_token_id" : vocab_dict["<s>"],
            "eos_token_id" : vocab_dict["</s>"],
            "pad_token_id" : vocab_dict[data_args.pad_token]
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
    )

    # Freeze encoder
    if model_args.freeze_feature_extractor:
        model.freeze_feature_extractor()

    # Create a single processor
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: load_metric(metric) for metric in data_args.eval_metrics}

    def compute_metrics(pred):
        # Prediction ids
        pred_ids = np.argmax(pred.predictions, axis=-1)
        # Convert -100 back to padding token id
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        # Prediction and label strings
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        # Calcualte the metrics
        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}
        return metrics

    # Load the dataset from your local files.
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    datasets = load_dataset("csv", data_files=data_files,
                            delimiter=data_args.delimiter, cache_dir=model_args.cache_dir)

    # Function to load audio and resample
    def load_audio_and_resample(path):
        # Load audio
        audio_array, sampling_rate = torchaudio.load(path)
        # Resample
        resampler = torchaudio.transforms.Resample(sampling_rate, sampling_rate_target)
        audio_array = resampler(audio_array).squeeze().numpy()
        return audio_array

    # Function to prepare inputs & targets
    def prepare_dataset(sample):
        # Load audio
        audio_array = load_audio_and_resample(sample[data_args.audio_path_column_name])
        # Input features
        inputs = feature_extractor(audio_array, sampling_rate=sampling_rate_target)
        sample["input_values"] = inputs.input_values[0]
        sample["input_length"] = len(sample["input_values"])
        # Encode targets
        sample["labels"] = tokenizer(sample[data_args.text_column_name]).input_ids
        return sample

    # Max & min input length for sample rate & max duration
    max_input_length = data_args.max_duration_in_seconds * sampling_rate_target
    min_input_length = data_args.min_duration_in_seconds * sampling_rate_target

    with training_args.main_process_first(desc="dataset map preprocessing"):
        # Prepare input features and targets
        datasets = datasets.map(
            prepare_dataset,
            remove_columns=[data_args.audio_path_column_name, data_args.text_column_name],
            desc="preprocess datasets",
            num_proc=data_args.preprocessing_num_workers
        )

        # Filter data samples based on length
        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        datasets = datasets.filter(
            is_audio_in_length_range,
            num_proc=data_args.preprocessing_num_workers,
            input_columns=["input_length"],
        )
        
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"] if data_args.validation_file else None,
        tokenizer=feature_extractor,
    )

    # Training
    if training_args.do_train:
        # Use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # Train metrics
        train_metrics = train_result.metrics
        train_metrics["train_samples"] = len(datasets["train"])
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_metrics = trainer.evaluate()
        # Evaluation metrics
        eval_metrics["eval_samples"] = len(datasets["validation"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

if __name__ == "__main__":
    main()
