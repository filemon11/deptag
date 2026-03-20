from . import settings, data, extraction
# Test
import sys

from typing import Collection

SETTINGS_NAME = "default"


def extract_multiple(
        settings: Collection[settings.Settings]
        ) -> extraction.Statistics:
    stats: None | extraction.Statistics = None
    for sett in settings:
        data_iterator = data.load_conllu(
            sett.file.conllu_file, sett.file.split,
        )

        stat = extraction.extract_and_write(
            data_iterator, sett.file.output_file,
            sett.deprels.arguments, sett.deprels.adjuncts,
            sett.deprels.delete, sett.deprels.merged,
            distinguish_fallback_subtypes=sett.deprels.subtypes,
            without_labels=not sett.deprels.labelled,
            merged_fallback_subtypes=sett.deprels.merged_fallback_subtypes,
            distinguish_merged_fallback_subtypes=(
                sett.deprels.distinguish_merged_fallback_subtypes),
            order_relations=sett.deprels.order_relations,
        )
        if stats is None:
            stats = stat
        else:
            stats += stat
    assert stats is not None
    return stats


if __name__ == "__main__":
    settings_name = SETTINGS_NAME
    if len(sys.argv) > 1:
        settings_name = sys.argv[1]

    # sett = settings.load_settings(
    #     settings_name=settings_name)
    # stat = extract_multiple([sett])
    # extraction.print_statistics(stat)

    setts1 = [
        settings.load_settings(settings_name=f"English-{name}")
        for name in (
            "Atis", "CHILDES", "EWT", "GUM",
            "LinES", "ParTUT", "GUMReddit", "ESLSpok") if not print(name)
    ]
    setts2 = [
        settings.load_settings(settings_name=f"French-{name}")
        for name in (
            "ALTS", "GSD", "ParisStories",
            "ParTUT", "Rhapsodie", "Sequoia") if not print(name)
    ]

    stat = extract_multiple(setts1) + extract_multiple(setts2)
    extraction.print_statistics(stat)
