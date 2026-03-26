from . import settings, data, extraction
# Test
import sys
import pathlib

from typing import Collection

SETTINGS_NAME = "default"


def load_and_write(
        sett: settings.Settings,
        *, replace_labels_in_unicorns: bool = False,
        temp_mode: bool = False,
        replacement_threshold: int = 1
        ):
    data_iterator = data.load_conllu(
        sett.file.conllu_file, sett.file.split,
    )

    output_file = sett.file.output_file
    if replace_labels_in_unicorns or temp_mode:
        output_file = f"{output_file}_temp"

    stat = extraction.extract_and_write(
        data_iterator, output_file,
        sett.deprels.arguments, sett.deprels.adjuncts,
        sett.deprels.delete, sett.deprels.merged,
        distinguish_fallback_subtypes=sett.deprels.subtypes,
        without_labels=not sett.deprels.labelled,
        merged_fallback_subtypes=sett.deprels.merged_fallback_subtypes,
        distinguish_merged_fallback_subtypes=(
            sett.deprels.distinguish_merged_fallback_subtypes),
        order_relations=sett.deprels.order_relations,
    )

    if replace_labels_in_unicorns and replacement_threshold > 0:
        unicorns = stat.unicorns
        if replacement_threshold > 1:
            unicorns = {
                supertag for supertag, num in stat.supertag_to_nums.items()
                if num <= replacement_threshold}
        data_iterator = data.load_conllu(
            output_file)
        return extraction.replace_unicorns_and_write(
            data_iterator, unicorns,
            sett.file.output_file)

    return stat


def extract_multiple(
        settings: Collection[settings.Settings],
        *, replace_labels_in_unicorns: bool = False,
        temp_mode: bool = False,
        replacement_threshold: int = 1,
        plot_unicorn_sentences: bool = False,
        ) -> extraction.Statistics:
    stats: None | extraction.Statistics = None
    for sett in settings:
        stat = load_and_write(
            sett, temp_mode=replace_labels_in_unicorns or temp_mode)
        if stats is None:
            stats = stat
        else:
            stats += stat
    assert stats is not None

    if ((replace_labels_in_unicorns and replace_labels_in_unicorns > 0)
            or plot_unicorn_sentences):
        new_stats: None | extraction.Statistics = None
        unicorns = stats.unicorns
        if replacement_threshold > 1:
            unicorns = {
                supertag for supertag, num in stats.supertag_to_nums.items()
                if num <= replacement_threshold}

        for sett in settings:
            data_iterator = data.load_conllu(
                f"{sett.file.output_file}_temp"
                if replace_labels_in_unicorns or temp_mode
                else sett.file.output_file)

            if plot_unicorn_sentences:
                data_iterator = iter(extraction.unicorn_plot_pipeline(
                    data_iterator, unicorns, pathlib.Path(
                        "plots", sett.file.conllu_file)
                ))

            if not (
                    replace_labels_in_unicorns
                    and replace_labels_in_unicorns > 0):
                for _ in iter(data_iterator):
                    pass
                continue

            stat = extraction.replace_unicorns_and_write(
                data_iterator, unicorns,
                sett.file.output_file)
            if new_stats is None:
                new_stats = stat
            else:
                new_stats += stat
        if new_stats is not None:
            return new_stats
    return stats


if __name__ == "__main__":
    settings_name = SETTINGS_NAME
    if len(sys.argv) > 1:
        settings_name = sys.argv[1]

    # sett = settings.load_settings(
    #     settings_name=settings_name)
    # setts = [
    #     settings.load_settings(settings_name=f"English-{name}")
    #     for name in (
    #         "Atis", "CHILDES") if not print(name)]
    # stat1 = extract_multiple(setts)
    # stat2 = extract_multiple(
    #     setts, replace_labels_in_unicorns=True,
    #     replacement_threshold=1)
    # extraction.print_statistics(stat1)
    # extraction.print_statistics(stat2)

    # setts1 = [
        # settings.load_settings(settings_name=f"English-{name}")
        # for name in (
        #     "EWT",
        #     "Atis", "CHILDES", "GUM",
        #     "LinES", "ParTUT", "GUMReddit", "ESLSpok"
        #     ) if not print(name)
    # ]
    # setts2 = [
    #     settings.load_settings(settings_name=f"French-{name}")
    #     for name in (
    #         "GSD",
    #         "ALTS", "ParisStories",
    #         "ParTUT", "Rhapsodie", "Sequoia"
    #         ) if not print(name)
    # ]

    # stat = extract_multiple(
    #     setts1 + setts2, replace_labels_in_unicorns=False,
    #     replacement_threshold=0, plot_unicorn_sentences=True)
    # TODO: put replace_labels in unicorns and replacement_threshold in
    # meta settings

    stat_dict: dict[str, extraction.Statistics] = {}
    for corpus in ("English-EWT", "English-EWT",):
        sett = settings.load_settings(settings_name=corpus)
        stat = extract_multiple(
            (sett,), replace_labels_in_unicorns=False,
            replacement_threshold=0, plot_unicorn_sentences=True)
        stat_dict[corpus] = stat
        extraction.print_statistics(stat)
