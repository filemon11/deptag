def reconstruct(
        simple_deprel: str, pos_tag: str, head_pos_tag: str) -> str:
    # TODO: Make this optional
    modifiers = (3,)
    nominals = ("PROPN", "NOUN", "PRON", "NUM", "SYM")
    predicates = ("VERB", "ADJ", "AUX")
    match simple_deprel:
        case "subj":
            if pos_tag in nominals:
                print("nsubj")
                return "nsubj"
            else:
                print("csubj")
                return "csubj"
        case "mod":
            if pos_tag == "ADV" and head_pos_tag in modifiers:
                print("advmod")
                return "advmod"
            elif pos_tag == "ADJ" and head_pos_tag in nominals:
                print("amod")
                return "amod"
            elif pos_tag == "ADV" and head_pos_tag in predicates:
                print("advcl")
                return "advcl"
            elif pos_tag == "NUM" and head_pos_tag in nominals:
                print("nummod")
                return "nummod"
            else:
                print("acl")
                return "acl"
        case _:
            return simple_deprel
