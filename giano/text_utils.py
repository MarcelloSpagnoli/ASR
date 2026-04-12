import string
import torch

CHARS = "-" + string.ascii_uppercase + " '?.!," # '-' è il BLANK (indice 0)
char_to_index = {char: idx for idx, char in enumerate(CHARS)}
index_to_char = {idx: char for idx, char in enumerate(CHARS)}

def text_to_int(text):
    return [char_to_index[c] for c in text.upper() if c in char_to_index]

def int_to_text(indices):
    return "".join([index_to_char[i] for i in indices])

def greedy_ctc_decode(log_probs):
    # log_probs shape: [T, C]
    best_path = torch.argmax(log_probs, dim=-1)
    decoded_text = []
    last_char_idx = -1

    for char_idx in best_path:
        idx = char_idx.item()
        if idx != 0 and idx != last_char_idx:
            decoded_text.append(index_to_char[idx])
        last_char_idx = idx
    return "".join(decoded_text)