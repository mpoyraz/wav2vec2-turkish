import re
from unicode_tr import unicode_tr

chars_to_remove_regex = '[,?.!\-\;\:"“%”�—…–()]'
apostrophes = "[’‘`´ʹʻʼʽʿˈ]"

def normalize_text(text):

    # Lower the text using 'unicode_tr'
    # Regular lower() does not work well for Turkish Language
    text_norm = unicode_tr(text).lower()
    # Unify apostrophes
    text_norm = re.sub(apostrophes, "'", text_norm)
    # Remove pre-defined chars
    text_norm = re.sub(chars_to_remove_regex, "", text_norm)
    # Remove single quotes
    text_norm = text_norm.replace(" '", " ")
    text_norm = text_norm.replace("' ", " ")
    # Handle hatted characters
    text_norm = re.sub('[â]', 'a', text_norm)
    text_norm = re.sub('[î]', 'i', text_norm)
    text_norm = re.sub('[ô]', 'o', text_norm)
    text_norm = re.sub('[û]', 'u', text_norm)
    # Handle alternate characters
    text_norm = re.sub('[é]', 'e', text_norm)
    text_norm = re.sub('[ë]', 'e', text_norm)
    # Remove multiple spaces
    text_norm = re.sub(r"\s+", " ", text_norm)

    return text_norm
