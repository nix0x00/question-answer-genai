from collections import Counter

def tokens_to_ids(word2idx, tokens):
        return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

def build_vocabulary(df, is_glove_frozen=True):
    """
    is_glove_frozen must be False if later trainable is set
    to True during layers.Embedding() process.
    """
    counter = Counter()

    for tokens in df["context_tokens"]:
        counter.update(tokens)
    
    for tokens in df["question_tokens"]:
        counter.update(tokens)
    
    MIN_FREQ = 2

    word2idx = { "<PAD>": 0, "<UNK>": 1 }

    for word, count in counter.most_common():
        if (is_glove_frozen):
             if count >= MIN_FREQ:
                  word2idx[word] = len(word2idx)
        else:
             word2idx[word] = len(word2idx)
    
    idx2word = {idx:word for word,idx in word2idx.items()}

    return word2idx, idx2word

def apply_vocabulary(df, word2idx):
     df["context_ids"] = df["context_tokens"].apply(lambda x: tokens_to_ids(word2idx, x))
     df["question_ids"] = df["question_tokens"].apply(lambda x: tokens_to_ids(word2idx, x))

     return df

def pad_sequences(series, size, pad_id = 0):
    def fix(row):
          row = row[:size]
          return row + [pad_id] * (size - len(row))
          
    return series.apply(fix)

def make_mask(series, size):
    return series.apply(lambda row: [1] * min(len(row), size)
                                    + [0] * max(0, size - len(row)))