def reconstruct(
        simple_deprel: str, pos_tag: str, head_pos_tag: str) -> str:
    modifiers = (3,)
    nominals = ("PROPN", "NOUN", "PRON", "NUM", "SYM")
    predicates = ("VERB", "ADJ", "AUX")
    match simple_deprel:
        case "subj":
            if pos_tag in nominals:
                return "nsubj"
            else:
                return "csubj"
        case "mod":
            if pos_tag == "ADV" and head_pos_tag in modifiers:
                return "advmod"
            elif pos_tag == "ADJ" and head_pos_tag in nominals:
                return "amod"
            elif pos_tag == "ADV" and head_pos_tag in predicates:
                return "advcl"
            elif pos_tag == "NUM" and head_pos_tag in nominals:
                return "nummod"
            else:
                return "acl"
        case _:
            return simple_deprel
