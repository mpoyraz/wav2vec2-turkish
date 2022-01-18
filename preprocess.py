import os
import re
import json
import argparse
import pandas as pd
from utils import remove_special_characters, unify_characters, check_invalid_char

def load_common_voice_corpus(path):

    # Load CommonVoice TSV files
    columns_keep = ['path', 'sentence']
    df_validated = pd.read_csv(os.path.join(path, 'validated.tsv'), sep='\t')[columns_keep]
    df_dev = pd.read_csv(os.path.join(path, 'dev.tsv'), sep='\t')[columns_keep]
    df_test = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t')[columns_keep]

    # Train set = Validated - (Dev + Test)
    dev_test_paths = df_dev['path'].to_list() + df_test['path'].to_list()
    df_train = df_validated[~df_validated['path'].isin(dev_test_paths)].copy()

    # Add full paths for audio records in train & dev splits
    df_train['path'] = df_train['path'].apply(lambda x: os.path.join(path,'clips',x))
    df_dev['path'] = df_dev['path'].apply(lambda x: os.path.join(path,'clips',x))

    return df_train, df_dev

def load_media_speech_corpus(path):

    # Load Media Speech corpus
    ms_paths, ms_sentences = [], []
    for files in os.listdir(path):
        if files.endswith('txt'):
            ms_paths.append(os.path.join(path, files.replace('txt','flac')))
            with open(os.path.join(path, files)) as fp:
                ms_sentences.append(fp.read().strip())

    df_ms = pd.DataFrame({'path': ms_paths, 'sentence': ms_sentences})
    return df_ms

def filter_dataset(df, vocab):
    # Clean and unify characters
    df['sentence'] = df['sentence'].apply(lambda x: unify_characters(remove_special_characters(x.lower())))
    # Replace multiple spaces with single space
    df['sentence'] = df['sentence'].apply(lambda x: re.sub('\s+',' ',x))
    # Keep samples with valid sentences only
    df['isInvalid'] = df['sentence'].apply(check_invalid_char, vocab=vocab)
    df = df[df['isInvalid']==0].drop(columns=['isInvalid'])
    return df

def load_dataset(path_cv_corpus, path_media_speech, vocab):
    # Load CommonVoice dataset
    df_cv_train, df_cv_dev = load_common_voice_corpus(path_cv_corpus)
    # Load MediaSpeech dataset
    df_ms = load_media_speech_corpus(path_media_speech)
    # Clean and filter datasets
    df_train = filter_dataset(pd.concat([df_cv_train, df_ms], ignore_index=True), vocab)
    df_dev = filter_dataset(df_cv_dev, vocab)
    return df_train, df_dev

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Wav2vec ASR with CTC")
    parser.add_argument("cv_path", type=str, help="Path for CommonVoice TR dataset")
    parser.add_argument("media_speech_path", type=str, help="Path for MediaSPeech TR dataset")
    parser.add_argument("vocab", type=str, help="ASR vocabulary of tokens")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    # Load vocab
    with open(args.vocab) as fp:
        vocab_dict = json.load(fp)

    # Load CommonVoice dataset
    df_cv_train, df_cv_dev = load_common_voice_corpus(args.cv_path)

    # Load MediaSpeech dataset
    df_ms = load_media_speech_corpus(args.media_speech_path)

    # Clean and filter datasets
    df_train = filter_dataset(pd.concat([df_cv_train, df_ms], ignore_index=True), vocab_dict)
    df_dev = filter_dataset(df_cv_dev, vocab_dict)

    # Save
    if args.output:
        df_train.to_csv(os.path.join(args.output, 'train.csv'), index=False)
        df_dev.to_csv(os.path.join(args.output, 'validation.csv'), index=False)

    return

if __name__ == "__main__":
    main()