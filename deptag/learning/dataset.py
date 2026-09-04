import json
import os

# import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from ..import extraction, data

from typing import (
    Mapping, Sequence, Iterable, Literal, Callable, TypeVar, Hashable,
    Type, Self)


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


K = TypeVar("K", bound=Hashable)


class Lexicon(dict[K, int]):
    unk: K

    def __init__(self, mapping=None, /, **kwargs):
        super().__init__(mapping, **kwargs)

    @classmethod
    def init(cls: Type[Self], mapping: Mapping[K, int], unknown: K) -> Self:
        lexicon = cls(mapping)
        lexicon[unknown] = len(lexicon)
        lexicon.unk = unknown
        return lexicon

    @classmethod
    def load(cls: Type[Self], mapping: Mapping[K, int], unknown: K) -> Self:
        lexicon = cls(mapping)
        lexicon.unk = unknown
        return lexicon

    def __getitem__(self, key: K) -> int:
        if not hasattr(self, "unk"):
            raise Exception("Use .init for initialisation.")

        if key not in self.keys():
            return self[self.unk]
        return super().__getitem__(key)


class UnkLexicon(Lexicon[str]):
    unk = "UNK"

    @classmethod
    def unk_init(cls: Type[Self], mapping: Mapping[str, int]) -> Self:
        lexicon = cls(mapping)
        lexicon[cls.unk] = len(lexicon)
        return lexicon

    @classmethod
    def unk_load(cls: Type[Self], mapping: Mapping[str, int]) -> Self:
        lexicon = cls(mapping)
        return lexicon


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
            self,
            split,
            tokenizer,
            tag_system: Mapping[str, int],
            data: Sequence[Sequence[extraction.Token]],
            dataset: str,
            max_train_len=350,
            factorised_max_left_right: None | tuple[int, int] = None,
            fraction: float = 1.0):
        self.fraction = fraction
        self.split = split
        self.trees = data
        self.tokenizer: transformers.PreTrainedTokenizerBase = tokenizer
        if not self.tokenizer.is_fast:
            raise TypeError(
                "TaggingDataset requires a fast tokenizer for word alignment."
            )
        self.dataset = dataset
        self.tag_system = dict(tag_system)
        self.id2sup = {i: sup for sup, i in tag_system.items()}
        self.id2sup_relative = {
            i: extraction.convert_string_to_relative_relation(sup)
            for i, sup in self.id2sup.items()}
        self.id2relative_sup = {
            i: extraction.process_relative_tag_to_projective(
                extraction.convert_string_to_relative_relation(sup))
            for i, sup in self.id2sup.items()}
        self.lr_args = [
            extraction.get_lr_argnum(tag)
            for tag in self.id2relative_sup.values() if tag is not None]
        self.max_l = max([lr[0] for lr in self.lr_args])
        self.max_r = max([lr[1] for lr in self.lr_args])

        self.pad_token_id = self.tokenizer.pad_token_id

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

        self.pos_dict = self._get_dict(
            "pos", dataset, split, self._create_dict_func("upos"))
        self.deprel_dict = self._get_dict(
            "deprel", dataset, split, self._create_dict_func(
                "deprel", self.deprel_to_main))
        self.sup_deprel_dict = self._get_dict(
            "sup_deprel", dataset, split, self._create_dict_func(
                "sup_deprel", self.deprel_to_main))
        self.xpos_dict = self._get_dict(
            "xpos", dataset, split, self._create_dict_func("xpos"))

        self.subtypes_dicts: dict[str, UnkLexicon] = {}
        for deprel in self.deprel_dict.keys():
            d_dict = self._get_dict(
                f"deprel_{deprel}", dataset, split,
                self._create_dict_func(
                    "deprel",
                    lambda x: (
                        self.deprel_to_sub(x)
                        if self.deprel_to_main(x) == deprel
                        else None)))
            if len(d_dict) > 2:
                self.subtypes_dicts[f"deprel_{deprel}"] = d_dict

        # feats dicts
        # Conditions only on the tokens that receive the feature
        # i.e. cannot be used to generate the features for a corpus;
        # only an auxiliary task
        self.feats_dicts: dict[str, UnkLexicon] = {}
        filename: str = f"./data/dicts/feats.{dataset.lower()}.json"
        if not os.path.exists(filename) and "train" in split:
            feats = set([
                key for sentence in self.trees
                for token in sentence
                for key in token.feats.keys()])

            for feature in feats:
                feat_dict: dict[str, int] = {}
                for sent in self.trees:
                    for token in sent:
                        if feature in token.feats:
                            feature_class = token.feats[feature]
                            feat_dict[feature_class] = feat_dict.get(
                                feature_class, len(feat_dict))
                        # else:
                        #     feat_dict["NOFEAT"] = feat_dict.get(
                        #         "NOFEAT", len(feat_dict))

                # Do not include feature if it has only one class
                # (ignoring UNK)
                if len(feat_dict) > 2:
                    self.feats_dicts[feature] = UnkLexicon.unk_init(feat_dict)
            with open(
                    filename, 'w') as fp:
                json.dump(self.feats_dicts, fp)
        else:
            with open(
                    filename, 'r') as fp:
                feats_dicts = json.load(fp)
                for name, mapping in feats_dicts.items():
                    self.feats_dicts[name] = UnkLexicon.unk_load(
                        mapping
                    )

        self.id2pos = {
            i: pos for pos, i in self.pos_dict.items()}
        self.id2deprel = {
            i: deprel for deprel, i in self.deprel_dict.items()}

    @property
    def sup2id(self) -> dict[str, int]:
        return self.tag_system

    @staticmethod
    def _get_dict(
            name: str, dataset: str,
            split: Literal["train", "dev", "test"],
            dict_getter: Callable[[], UnkLexicon]) -> UnkLexicon:

        filename: str = f"./data/dicts/{name}.{dataset.lower()}.json"
        if not os.path.exists(
                filename
                ) and "train" in split:
            out_dict = dict_getter()
            with open(
                    filename, 'w') as fp:
                json.dump(out_dict, fp)
        else:
            with open(
                    filename, 'r') as fp:
                out_dict = json.load(fp)
                out_dict = UnkLexicon.unk_load(out_dict)
                # This should be made adaptable for other UNK tokens
        return out_dict

    def _create_dict_func(
            self,
            token_attr: str,
            transformation: None | Callable[[str | None], str | None] = None
            ) -> Callable[[], UnkLexicon]:

        def getter() -> UnkLexicon:
            out_dict: dict[str, int] = {}
            for sent in self.trees:
                for token in sent:
                    x = getattr(token, token_attr)
                    if transformation is not None:
                        x = transformation(x)
                    if x is not None:
                        out_dict[x] = out_dict.get(
                            x, len(out_dict))
            return UnkLexicon.unk_init(out_dict)
        return getter

    @staticmethod
    def deprel_to_main(deprel: str | None) -> str | None:
        if deprel is None:
            return None
        if data.has_subtype(deprel):
            return data.split_main_sub(deprel)[0]
        return deprel

    @staticmethod
    def deprel_to_sub(deprel: str | None) -> str | None:
        if deprel is None:
            return deprel
        if data.has_subtype(deprel):
            return data.split_main_sub(deprel)[1]
        return "NOSUB"

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

    def __len__(self):
        return int(len(self.trees) * self.fraction)  # /24)  # TODO

    def __getitem__(self, index: int):
        sent = self.trees[int(index*(1/self.fraction))]
        words = ptb_unescape(w.word for w in sent)

        words = [w.replace("\xad", "") for w in words]
        # necessary to remove soft-hyphens from Romanian RRT dataset

        heads: torch.Tensor = torch.tensor(
            [word.head for word in sent], dtype=torch.long)  # - 1
        # use BOS token as root

        pos_tags = [self.pos_dict[w.upos] for w in sent]
        xpos_tags = [self.xpos_dict[w.xpos] for w in sent]
        deprel_tags = [self.deprel_dict[
            data.split_main_sub(
                w.deprel)[0] if data.has_subtype(w.deprel) else w.deprel]
                for w in sent]
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
            (
                self.tag_system[w.sup] if w.sup in self.tag_system
                else self.tag_system["-UNK*"])
            for w in sent]

        tag_ids = torch.tensor(tag_ids_, dtype=torch.long)
        # labels = torch.full_like(input_ids, -1)

        factorised_tags = [
            extraction.convert_relative_tag_to_factorised(
                extraction.convert_string_to_relative_relation(w.sup))
            for w in sent]

        factorised_dict = dict()
        if self.max_left is not None and self.max_right is not None:
            l_arg_nums = [tag[0] for tag in factorised_tags]
            l_args = [
                list(reversed([None] * (
                    self.max_left - len(tag[1])) + tag[1]))
                for tag in factorised_tags]
            l_arg_ids = [
                [(self.sup_deprel_dict[deprel]
                    if deprel is not None else -1)
                    for deprel in word]
                for word in l_args]
            # TODO: these must be passed individually
            r_arg_nums = [tag[2] for tag in factorised_tags]
            r_args = [
                tag[3] + [None] * (self.max_right - len(tag[3]))
                for tag in factorised_tags]
            r_arg_ids = [
                [(self.sup_deprel_dict[deprel]
                    if deprel is not None else -1)
                    for deprel in word]
                for word in r_args]
            aux_positions = [
                (tag[4] if tag[4] is not None else 0) + self.max_left + 1
                for tag in factorised_tags]
            # maximum is 2 + max_left + max_right

            aux_rel_ids = [
                (
                    self.sup_deprel_dict[tag[5]]
                    if tag[5] is not None else -1)
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

        feats_dict = {
            feat: torch.tensor(
                [
                    (
                        f_dict[w.feats[feat]]
                        if feat in w.feats else -1)  # f_dict["NOFEAT"])
                    for w in sent],
                dtype=torch.long)
            for feat, f_dict in self.feats_dicts.items()}

        subtypes_dict = {
            deprel: torch.tensor(
                [
                    (
                        self.subtypes_dicts[deprel][s]
                        if (d := self.deprel_to_main(w.deprel)) is not None
                        and (s := self.deprel_to_sub(w.deprel)) is not None
                        and deprel.endswith(d) else -1)
                    for w in sent],
                dtype=torch.long)
            for deprel in self.subtypes_dicts.keys()}

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
            "xpos_ids": torch.tensor(
                xpos_tags,
                dtype=torch.long,
            ),
            "labels": tag_ids,
            "heads": heads,  # original UD heads: 0..num_words
            "deprel_ids": torch.tensor(
                deprel_tags,
                dtype=torch.long),
        } | factorised_dict | feats_dict | subtypes_dict

    def collate(self, batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = input_ids.ne(
            self.pad_token_id
        )

        keys = set(batch[0].keys()) - {"input_ids"}

        # -1 means that this padded slot contains no word.
        output_dict = {
            key: pad_sequence(
                [item[key] for item in batch],
                batch_first=True,
                padding_value=-1,
            )
            for key in keys
        } | {
            "input_ids": input_ids, "attention_mask": attention_mask}

        return output_dict
