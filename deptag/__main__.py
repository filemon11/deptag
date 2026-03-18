from . import settings, data, extraction
# Test
UD_NAME = "English-GUM"

sett = settings.load_settings(
    standard_name=UD_NAME,
    standard_from_xml=True,
)

data_iterator = data.load_conllu(
    UD_NAME, "train",
)

ext = extraction.extract_and_write(
    data_iterator, "test", sett.arguments, sett.adjuncts, sett.delete,
    distinguish_fallback_subtypes=True,
    without_labels=True
)