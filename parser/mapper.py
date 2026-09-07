def map_char_to_token(row):
    start_token, end_token = -1, -1
    offsets = row["context_offsets"]
    ans_start = row["answer_start"]
    ans_end = row["answer_end"]

    for idx, (s, e) in enumerate(offsets):
        if (start_token == -1 and ans_start >= s and ans_start < e):
            start_token = idx

        if (ans_end > s and ans_end <= e):
            end_token = idx

        if (start_token != -1 and end_token != -1):
            break

    return start_token, end_token

def map_answer_to_positions(encodings, df):
    sample_mapping = encodings["overflow_to_sample_mapping"]
    offset_mapping = encodings["offset_mapping"]

    # Pull once — no per-iteration Series construction
    answer_starts = df["answer_start"].to_numpy()
    answer_ends   = df["answer_end"].to_numpy()

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        a_start = answer_starts[sample_idx]
        a_end   = answer_ends[sample_idx]

        seq_ids = encodings.sequence_ids(i)   # None=special, 0=question, 1=context

        # Locate the context span
        try:
            ctx_start = seq_ids.index(1)
        except ValueError:
            start_positions.append(0)
            end_positions.append(0)
            continue

        ctx_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)

        # Answer fully outside this feature's context window -> CLS (0,0)
        if offsets[ctx_start][0] > a_start or offsets[ctx_end][1] < a_end:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_idx = ctx_start
        while token_idx <= ctx_end and offsets[token_idx][0] <= a_start:
            token_idx += 1
        start_positions.append(token_idx - 1)

        token_idx = ctx_end
        while token_idx >= ctx_start and offsets[token_idx][1] >= a_end:
            token_idx -= 1
        end_positions.append(token_idx + 1)

    encodings["start_positions"] = start_positions
    encodings["end_positions"] = end_positions

    return encodings