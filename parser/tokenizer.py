import re

def tokenize(df):

    def process_text(text):
        tokens, offsets = [], []
        for m in re.finditer(r"\w+|[^\w\s]", text):  # \S+ if only words need to be separated
            tokens.append(m.group())
            offsets.append((m.start(), m.end()))

        return (tokens, offsets)

    df["context_tokens"], df["context_offsets"] = zip(*df["context"].apply(process_text))
    df["question_tokens"] = df["question"].apply(lambda x: process_text(x)[0])

    return df