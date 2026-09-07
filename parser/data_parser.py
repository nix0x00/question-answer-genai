import pandas as pd


def parser(data, mode: str = "train"):
    titles, contexts, question_ids, questions, answer_starts, answer_ends, texts = (
        [],        [],        [],        [],        [],        [],        [],
    )

    for row in data:
        title = row.get("title", "")
        for paragraph in row.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qas in paragraph.get("qas", []):
                question = qas.get("question", "")
                id = qas.get("id", "")
                for answer in qas.get("answers", []):
                    answer_start = answer.get("answer_start", "")
                    text = answer.get("text", "")

                    # since we are using 2024 GloVe therefore, converted to lowercase
                    titles.append(title.lower())
                    contexts.append(context.lower())
                    question_ids.append(id)
                    questions.append(question.lower())
                    answer_starts.append(answer_start)
                    answer_ends.append(answer_start + len(text))
                    texts.append(text.lower())

    return pd.DataFrame(
        {
            "title": titles,
            "context": contexts,
            "question_id": question_ids,
            "question": questions,
            "answer_start": answer_starts,
            "answer_end": answer_ends,
            "text": texts,
        }
    )


def clean_dev_data(df):
    df = df.drop_duplicates(subset=["question_id", "text"])
    df = df.groupby("question_id", as_index=False).agg(
        {
            "title": "first",
            "context": "first",
            "question_id": "first",
            "question": "first",
            "context_tokens": "first",
            "context_offsets": "first",
            "question_tokens": "first",
            "answer_start": "first",
            "answer_end": "first",
            "text": list,
            "start_token": list,
            "end_token": list,
        }
    )

    return df


def clean_dev_encodings(encodings, df):
    bert_df = pd.DataFrame(
        {
            "question_id": df["question_id"],
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "text": df["text"],
            "start_positions": encodings["start_positions"],
            "end_positions": encodings["end_positions"],
        }
    )

    bert_df = (
        bert_df.drop_duplicates(subset=["question_id", "text"])
        .groupby("question_id", as_index=False, sort=False)
        .agg(
            {
                "input_ids": "first",
                "attention_mask": "first",
                "text": list,
                "start_positions": list,
                "end_positions": list,
            }
        )
    )


def filter_spans(df, c_len):
    # rules to guard against breaking and inverted spans
    m = (
        (df.start_token >= 0)
        & (df.end_token >= 0)
        & (df.start_token < c_len)
        & (df.end_token < c_len)
        & (df.start_token <= df.end_token)
    )
    
    return df[m].reset_index(drop=True)
