import json
import os

# import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from ..import extraction, data

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
            max_train_len=350,
            factorised_max_left_right: None | tuple[int, int] = None):
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

        if factorised_max_left_right is None:
            self.max_left = None
            self.max_right = None
        else:
            self.max_left, self.max_right = factorised_max_left_right

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
                pos_dict[x] = pos_dict.get(x, len(pos_dict))
        return pos_dict

    def get_deprel_dict(self):
        # TOOD: add option for subtypes
        deprel_dict: dict[str, int] = {}
        for sent in self.trees:
            for _, _, _, _, x in sent:
                if data.has_subtype(x):
                    x = data.split_main_sub(x)[0]
                deprel_dict[x] = deprel_dict.get(x, len(deprel_dict))
        return deprel_dict

    def __len__(self):
        return int(len(self.trees))  # /24)  # TODO

    def __getitem__(self, index: int):
        sent = self.trees[index]
        words = ptb_unescape(w[0] for w in sent)

        words = [w.replace("\xad", "") for w in words]
        # necessary to remove soft-hyphens from Romanian RRT dataset

        heads: torch.Tensor = torch.tensor(
            [word[3] for word in sent], dtype=torch.long)  # - 1
        # use BOS token as root

        pos_tags = [self.pos_dict.get(w[1], 0) for w in sent]
        deprel_tags = [self.deprel_dict.get(
            data.split_main_sub(w[4])[0] if data.has_subtype(w[4]) else w[4],
            0) for w in sent]

        # encoded = self.tokenizer._encode_plus(' '.join(words))
        # word_end_positions = [
        #     encoded.char_to_token(i)
        #     for i in np.cumsum([len(word) + 1 for word in words]) - 2]

        # input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)

        input_ids, word_end_positions = self._encode_words(words)

        end_of_word = torch.zeros_like(input_ids)
        # pos_ids = torch.full_like(input_ids, -1)
        # deprel_ids = torch.full_like(input_ids, -1)

        tag_ids_: list[int] = [
            (self.tag_system[w[2]] if w[2] in self.tag_system else 0)
            for w in sent]

        tag_ids = torch.tensor(tag_ids_, dtype=torch.long)
        # labels = torch.full_like(input_ids, -1)

        factorised_tags = [
            extraction.convert_relative_tag_to_factorised(
                extraction.convert_string_to_relative_relation(w[2]))
            for w in sent]

        factorised_dict = dict()
        if self.max_left is not None and self.max_right is not None:
            l_arg_nums = [tag[0] for tag in factorised_tags]
            l_args = [
                list(reversed([None] * (
                    self.max_left - len(tag[1])) + tag[1]))
                for tag in factorised_tags]
            l_arg_ids = [
                [(self.deprel_dict.get(
                    deprel, 0) if deprel is not None else -1)
                    for deprel in word]
                for word in l_args]
            # TODO: these must be passed individually
            r_arg_nums = [tag[2] for tag in factorised_tags]
            r_args = [
                tag[3] + [None] * (self.max_right - len(tag[3]))
                for tag in factorised_tags]
            r_arg_ids = [
                [(self.deprel_dict.get(
                    deprel, 0) if deprel is not None else -1)
                    for deprel in word]
                for word in r_args]
            aux_positions = [
                (tag[4] if tag[4] is not None else 0) + self.max_left + 1
                for tag in factorised_tags]
            # maximum is 2 + max_left + max_right

            aux_rel_ids = [
                (self.deprel_dict.get(tag[5], 0) if tag is not None else -1)
                for tag in factorised_tags]

            left_dict = {
                f"left_{i}": torch.tensor(
                    [word[i-1] for word in l_arg_ids], dtype=torch.long
                ) for i in range(1, self.max_left+1)
            }
            right_dict = {
                f"right_{i}": torch.tensor(
                    [word[i-1] for word in r_arg_ids], dtype=torch.long
                ) for i in range(1, self.max_right+1)
            }

            factorised_dict = {
                "l_arg_nums": torch.tensor(
                    l_arg_nums, dtype=torch.long),
                "r_arg_nums": torch.tensor(
                    r_arg_nums, dtype=torch.long
                ),
                "aux_positions": torch.tensor(
                    aux_positions, dtype=torch.long
                ),
                "aux_rel_ids": torch.tensor(
                    aux_rel_ids, dtype=torch.long
                ),
            } | left_dict | right_dict

        # heads_long = torch.full_like(input_ids, -1)

        # labels[word_end_positions] = tag_ids
        # pos_ids[word_end_positions] = torch.tensor(
        #   pos_tags, dtype=torch.long)
        # deprel_ids[word_end_positions] = torch.tensor(
        #    # deprel_tags, dtype=torch.long)
        # # print("heads", heads)
        # heads_long[word_end_positions] = heads  # [-1, x, y, -1, z]
        # # print("heads_long1", heads_long)
        # cumsum = torch.cumsum(heads_long == -1, dim=0) - 1
        # # [0, (0, 0), 1, (1,)]
        # # print("cumsum", cumsum)
        # head_cumsum = cumsum[~(heads_long == -1)]
        # # [1, (1, 1), 2, (2,)][False, True, True, False, True] -1 = [0, 0, 1]
        # head_cumsum = head_cumsum[heads-1 + ((heads - 1) < 0)]
        # head_cumsum[heads == 0] = 0
        # # [h_c[x], h_c[y], h_c[z]]

        # # print("head_cumsum", head_cumsum)

        # heads_long[word_end_positions] += head_cumsum
        # print("heads_long", heads_long)
        # print("pos_ids", pos_ids)
        end_of_word[word_end_positions] = 1
        end_of_word[word_end_positions[-1]] = 2  # last word

        # return {
        #     'input_ids': input_ids,
        #     'pos_ids': pos_ids,
        #     'end_of_word': end_of_word,
        #     'labels': labels,
        #     'heads': heads_long,
        #     'deprel_ids': deprel_ids,
        # }
        return {
            "input_ids": input_ids,

            # Transformer positions of the final subword of each UD word.
            # Shape: [num_words]
            "word_end_positions": word_end_positions,

            # All of these are now WORD-level.
            "pos_ids": torch.tensor(
                pos_tags,
                dtype=torch.long,
            ),
            "labels": tag_ids,
            "heads": heads,  # original UD heads: 0..num_words
            "deprel_ids": torch.tensor(
                deprel_tags,
                dtype=torch.long),
        } | factorised_dict

    def collate(self, batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = input_ids.ne(
            self.pad_token_id
        )

        # -1 means that this padded slot contains no word.
        word_end_positions = pad_sequence(
            [item["word_end_positions"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        pos_ids = pad_sequence(
            [item["pos_ids"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        heads = pad_sequence(
            [item["heads"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        deprel_ids = pad_sequence(
            [item["deprel_ids"] for item in batch],
            batch_first=True,
            padding_value=-1,
        )

        output_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,

            "word_end_positions": word_end_positions,

            "pos_ids": pos_ids,
            "labels": labels,
            "heads": heads,
            "deprel_ids": deprel_ids,
        }

        if self.max_left is not None and self.max_right is not None:

            output_dict |= {
                "l_arg_nums": pad_sequence(
                    [item["l_arg_nums"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
                "r_arg_nums": pad_sequence(
                    [item["r_arg_nums"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
                "aux_positions": pad_sequence(
                    [item["aux_positions"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
                "aux_rel_ids": pad_sequence(
                    [item["aux_rel_ids"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ),
            } | {
                f"left_{i}": pad_sequence(
                    [item[f"left_{i}"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ) for i in range(1, self.max_left+1)
            } | {
                f"right_{i}": pad_sequence(
                    [item[f"right_{i}"] for item in batch],
                    batch_first=True,
                    padding_value=-1,
                ) for i in range(1, self.max_right+1)
            }

        return output_dict
