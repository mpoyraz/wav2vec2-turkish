import re

def remove_special_characters(text, chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\'\(\)…]'):
    text = re.sub(chars_to_remove_regex, '', text)
    return text

def unify_characters(text):
    # Hatted characters
    text = re.sub('[â]', 'a', text)
    text = re.sub('[î]', 'i', text)
    text = re.sub('[ô]', 'o', text)
    text = re.sub('[û]', 'u', text)
    # Alternate characters
    text = re.sub('[é]', 'e', text)
    text = re.sub('[ë]', 'e', text)
    text = re.sub('[i̇]', 'i', text)
    # Apostrophe
    text = re.sub('[’]', "'", text)
    return text

def check_invalid_char(sentence, vocab):
    return any([ch not in vocab for ch in re.sub(r"\s+", "", sentence)])
