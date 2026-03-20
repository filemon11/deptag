from . import settings, standards
from .. import data


def assert_dep_settings(
        sett: settings.DepSettings) -> None:
    if sett.merged is not None:
        for i, (a_name, a_list) in enumerate(sett.merged.items()):
            for b_name, b_list in list(sett.merged.items())[i+1:]:
                assert len(set(a_list).intersection(b_list)) == 0, (
                    "Duplicate relation in definitions of merged"
                    f" deprels '{a_name}' and '{b_name}'."
                )


def assert_dep_standard(
        sett: settings.DepSettings,
        stan: standards.DeprelStandard,
        *,
        allow_partial_underspecification: bool = True,
        ) -> None:
    """Checks if all deprels and subtypes
    mentioned in settings are defined in standard
    and if all deprels defined in standard are
    mentioned in settings

    Parameters
    ----------
    sett : settings.DepSettings
        _description_
    stan : standards.DeprelStandard
        _description_

    Returns
    -------
    bool
        _description_
    """
    # Check if deprel sets in settings are disjoint
    assert set(sett.adjuncts).isdisjoint(set(sett.arguments)), (
        "Deprel types adjuncts and arguments are not disjoint"
    )
    assert set(sett.arguments).isdisjoint(set(sett.delete)), (
        "Deprel types arguments and delete are not disjoint"
    )
    assert set(sett.delete).isdisjoint(set(sett.adjuncts)), (
        "Deprel types delete and adjuncts are not disjoint"
    )

    # Check if deprels mentioned in settings are defined
    # in standard
    subtype: None | str
    for deprel in sett.arguments + sett.adjuncts + sett.delete:
        subtype = None
        if data.has_subtype(deprel):
            deprel, subtype = data.split_main_sub(deprel)

        assert deprel in stan.labels, (
            f"Deprel '{deprel}' is not defined in standard."
        )

        if subtype is not None:
            assert subtype in stan.labels[deprel], (
                f"Subtype '{subtype}' is not defined for "
                f"deprel '{deprel}' in standard"
            )

    # Mark for which deprel settings specified at least one subtype.
    # Then, all subtypes defined in standard must for this deprel must
    # appear explicitly in settings

    # -> Convert settings deprels into standard format
    sett_deprels: dict[str, list[str]] = {}
    for deprel in sett.arguments + sett.adjuncts + sett.delete:
        subtype = None
        if data.has_subtype(deprel):
            deprel, subtype = data.split_main_sub(deprel)
        if deprel not in sett_deprels:
            sett_deprels[deprel] = []
        if subtype is not None:
            sett_deprels[deprel].append(subtype)

    for stan_deprel, subtypes in stan.labels.items():
        assert stan_deprel in sett_deprels, (
            f"Deprel '{stan_deprel}' defined in standard is not"
            " mentioned in settings."
        )
        if (
                not allow_partial_underspecification
                and len(subtypes) > 0
                and len(sett_deprels[stan_deprel]) > 0):
            for subt in subtypes:
                assert subt in sett_deprels[stan_deprel], (
                    f"Standard subtype '{subt}' for deprel '{stan_deprel}' "
                    "is not mentioned "
                    "in settings despite more than one subtype being "
                    f"specified there: ({sett_deprels[stan_deprel]})"
                )
