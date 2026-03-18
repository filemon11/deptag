def has_subtype(deprel: str) -> bool:
    return ":" in deprel


def assert_has_subtype(deprel: str) -> None:
    if not has_subtype(deprel):
        raise TypeError(f"Deprel '{deprel}' has no subtype.")


def split_main_sub(deprel: str) -> tuple[str, str]:
    assert_has_subtype(deprel)
    splits = deprel.split(":")
    if len(splits) > 2:
        raise TypeError(f"Deprel '{deprel}' has more than two components.")
    return splits  # type: ignore
