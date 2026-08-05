import json
import os

# import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers

from typing import Mapping, Sequence, Iterable


PTB_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


def ptb_unescape(sent: Iterable[str]) -> list[str]:
    cleaned_words: list[str] = []
    for word in sent:
        word = PTB_TOKEN_MAPPING.get(word, word)
        word = word.replace('\\/', '/').replace('\\*', '*')
        # Mid-token punctuation occurs in biomedical text
        word = word.replace('-LSB-', '[').replace('-RSB-', ']')
        word = word.replace('-LRB-', '(').replace('-RRB-', ')')
        if word == "n't" and cleaned_words:
            cleaned_words[-1] = cleaned_words[-1] + "n"
            word = "'t"
        cleaned_words.append(word)
    return cleaned_words


class TaggingDataset(torch.utils.data.Dataset):
    def __init__(
            self, split, tokenizer, tag_system: Mapping[str, int],
            data: Sequence[Sequence[tuple[str, str, str, int, str]]], device,
            dataset: str,
            max_train_len=350):
        self.split = split
        self.trees = data
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer
        if not self.tokenizer.is_fast:
            raise TypeError(
                "TaggingDataset requires a fast tokenizer for word alignment."
            )
        self.dataset = dataset
        self.tag_system = tag_system
        self.pad_token_id = self.tokenizer.pad_token_id
        self.device = device

        if "train" in split and max_train_len is not None:
            # To speed up training, we only train on short sentences.
            print(len(self.trees), f"sentences from {split} before filtering")
            self.trees = [
                sent for sent in self.trees if (
                    len(sent) <= max_train_len
                    and len(sent) >= 2)]
            print(len(self.trees), f"trees from {split} after filtering")
        else:
            # speed up!
            self.trees = [
                sent for sent in self.trees
                if len(sent) <= max_train_len]
            print(f"Loaded {len(self.trees)} sentences from {split}")

        if not os.path.exists(
                f"./data/pos/pos.{dataset.lower()}.json"
                ) and "train" in split:
            self.pos_dict = self.get_pos_dict()
            with open(f"./data/pos/pos.{dataset.lower()}.json", 'w') as fp:
                json.dump(self.pos_dict, fp)
        else:
            with open(f"./data/pos/pos.{dataset.lower()}.json", 'r') as fp:
                self.pos_dict = json.load(fp)

        if not os.path.exists(
                f"./data/deprel/deprel.{dataset.lower()}.json"
                ) and "train" in split:
            self.deprel_dict = self.get_deprel_dict()
            with open(
                    f"./data/deprel/deprel.{dataset.lower()}.json", 'w') as fp:
                json.dump(self.deprel_dict, fp)
        else:
            with open(
                    f"./data/deprel/deprel.{dataset.lower()}.json", 'r') as fp:
                self.deprel_dict = json.load(fp)

    def _encode_words(
            self,
            words: Sequence[str],
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode words and return final-subword positions.

        Returns
        -------
        input_ids
            Token IDs including the initial and final special tokens.
        word_end_positions
            For each input word, the position of its final subword in
            input_ids.
        """
        encoded = self.tokenizer(
            list(words),
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        input_ids = torch.tensor(
            encoded["input_ids"],
            dtype=torch.long,
        )

        word_ids = encoded.word_ids()

        if word_ids is None:
            raise RuntimeError(
                "The tokenizer did not return word alignment information."
            )

        if len(word_ids) != len(input_ids):
            raise RuntimeError(
                "word_ids and input_ids have different lengths."
            )

        # Both BERT and RoBERTa/XLM-R place their sentence-level special
        # token at position 0. This position is used as the dependency root.
        if word_ids[0] is not None:
            raise RuntimeError(
                "The first token is not a special token and therefore "
                "cannot be used as the artificial root."
            )

        if (
            self.tokenizer.cls_token_id is not None
            and input_ids[0].item() != self.tokenizer.cls_token_id
        ):
            raise RuntimeError(
                f"Expected initial token ID "
                f"{self.tokenizer.cls_token_id}, "
                f"got {input_ids[0].item()}."
            )

        # Repeated assignment retains the final subword of each word.
        word_end_positions = torch.full(
            (len(words),),
            fill_value=-1,
            dtype=torch.long,
        )

        for token_position, word_index in enumerate(word_ids):
            if word_index is not None:
                word_end_positions[word_index] = token_position

        missing = torch.nonzero(
            word_end_positions < 0,
            as_tuple=False,
        ).flatten()

        if len(missing) > 0:
            missing_indices = missing.tolist()
            missing_words = [words[i] for i in missing_indices]

            raise ValueError(
                f"The tokenizer produced no subwords for word indices "
                f"{missing_indices}: {missing_words}"
            )

        model_max_length = self.tokenizer.model_max_length

        # Some tokenizers use an extremely large sentinel value when there
        # is no known maximum. BERT/RoBERTa normally provide a real limit.
        has_real_limit = model_max_length < 1_000_000

        if has_real_limit and len(input_ids) > model_max_length:
            raise ValueError(
                f"Sentence produces {len(input_ids)} subwords, but "
                f"{self.tokenizer.name_or_path} supports at most "
                f"{model_max_length}."
            )

        return input_ids, word_end_positions

    def get_pos_dict(self):
        pos_dict: dict[str, int] = {}
        for sent in self.trees:
            for _, x, _, _, _ in sent:
                pos_dict[x] = pos_dict.get(x, 1 + len(pos_dict))
        return pos_dict

    def get_deprel_dict(self):
        deprel_dict: dict[str, int] = {}
        for sent in self.trees:
            for _, _, _, _, x in sent:
                deprel_dict[x] = deprel_dict.get(x, 1 + len(deprel_dict))
        return deprel_dict

    def __len__(self):
        return int(len(self.trees)/1)  # 24)

    def __getitem__(self, index: int):
        sent = self.trees[index]
        words = ptb_unescape(w[0] for w in sent)

        words = [w.replace("\xad", "") for w in words]
        # necessary to remove soft-hyphens from Romanian RRT dataset

        heads: torch.Tensor = torch.tensor(
            [word[3] for word in sent], dtype=torch.long)  # - 1
        # use BOS token as root

        pos_tags = [self.pos_dict.get(w[1], 0) for w in sent]
        deprel_tags = [self.deprel_dict.get(w[4], 0) for w in sent]

        # encoded = self.tokenizer._encode_plus(' '.join(words))
        # word_end_positions = [
        #     encoded.char_to_token(i)
        #     for i in np.cumsum([len(word) + 1 for word in words]) - 2]

        # input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)

        input_ids, word_end_positions = self._encode_words(words)

        end_of_word = torch.zeros_like(input_ids)
        pos_ids = torch.full_like(input_ids, -1)
        deprel_ids = torch.full_like(input_ids, -1)

        tag_ids_: list[int] = [
            (self.tag_system[w[2]] if w[2] in self.tag_system else 0)
            for w in sent]

        tag_ids = torch.tensor(tag_ids_, dtype=torch.long)
        labels = torch.full_like(input_ids, -1)

        heads_long = torch.full_like(input_ids, -1)

        labels[word_end_positions] = tag_ids
        pos_ids[word_end_positions] = torch.tensor(pos_tags, dtype=torch.long)
        deprel_ids[word_end_positions] = torch.tensor(
            deprel_tags, dtype=torch.long)
        # print("heads", heads)
        heads_long[word_end_positions] = heads  # [-1, x, y, -1, z]
        # print("heads_long1", heads_long)
        cumsum = torch.cumsum(heads_long == -1, dim=0) - 1  # [0, (0, 0), 1, (1,)]
        # print("cumsum", cumsum)
        head_cumsum = cumsum[~(heads_long == -1)]
        # [1, (1, 1), 2, (2,)][False, True, True, False, True] -1 = [0, 0, 1]
        head_cumsum = head_cumsum[heads-1 + ((heads - 1) < 0)]
        head_cumsum[heads == 0] = 0
        # [h_c[x], h_c[y], h_c[z]]

        # print("head_cumsum", head_cumsum)

        heads_long[word_end_positions] += head_cumsum
        # print("heads_long", heads_long)
        # print("pos_ids", pos_ids)
        end_of_word[word_end_positions] = 1
        end_of_word[word_end_positions[-1]] = 2  # last word

        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'end_of_word': end_of_word,
            'labels': labels,
            'heads': heads_long,
            'deprel_ids': deprel_ids,
        }

    def collate(self, batch):
        # for GPT-2, self.pad_token_id is None
        # pad_token_id = (
        #     self.pad_token_id if self.pad_token_id is not None
        #     else -100)
        if self.pad_token_id is None:
            raise ValueError("The tokenizer has no padding token.")

        input_ids = pad_sequence(
            [item['input_ids'] for item in batch],
            batch_first=True, padding_value=self.pad_token_id)

        # attention_mask = (input_ids != pad_token_id).float()
        attention_mask = input_ids.ne(self.pad_token_id)

        # # for GPT-2, change -100 back into 0
        # input_ids = torch.where(
        #     input_ids == -100,
        #     0,
        #     input_ids
        # )

        end_of_word = pad_sequence(
            [item['end_of_word'] for item in batch],
            batch_first=True, padding_value=0)

        pos_ids = pad_sequence(
            [item['pos_ids'] for item in batch],
            batch_first=True, padding_value=-1)

        deprel_ids = pad_sequence(
            [item['deprel_ids'] for item in batch],
            batch_first=True, padding_value=-1)

        labels = pad_sequence(
            [item['labels'] for item in batch],
            batch_first=True, padding_value=-1)

        heads = pad_sequence(
            [item['heads'] for item in batch],
            batch_first=True, padding_value=-1)

        return {
            'input_ids': input_ids,
            'pos_ids': pos_ids,
            'deprel_ids': deprel_ids,
            'end_of_word': end_of_word,
            'attention_mask': attention_mask,
            'labels': labels,
            'heads': heads,
        }
