# DepTAG: Tag for dependency trees

The experiments were conducted using Universal Dependencies version 2.15 for comparability with previous parsers.

## Setup

TODO: add environment information
TODO: allow other UD versions

### Data import

#### Importing UD treebanks

1. Download Universal Dependencies (UD) version 2.15 at http://hdl.handle.net/11234/1-5787

2. Create a folder named "data" in the deptag directory and place the UD zip file inside it.

3. Unzip the UD file.

For other UD versions, follow the same procedure and specify the `ud_folder` argument in the settings file (see below).

#### Importing other treebanks

TODO

In order to prevent unwanted behaviour, it is necessary to specify a "standard" for your treebank. In the case of UD data, it is extracted automatically from the treebank's stats.xml file. For other data, a standards file in the standards folder must be used.

TODO

#### Combine treebanks

TODO

### Settings file

(Almost) all parameters for supertag extraction, parser training and evaluation are to be specified in a .toml settings file placed in the settings folder. You can simply copy the default.toml file, rename the new file to \<your_name\>.toml and modify it.

When running the program you will refer to your settings using the argument `--settings English-EWT`.

#### File settings

- conllu_file: the name of the treebank file; in the case of UD imports, the name of the treebank
- ud_folder: the UD directory; defaults to data/Universal Dependencies 2.15/ud-treebanks-v2.15
- data_folder: the directory whether the conllu_file is located, in the case of non-UD-imports; defaults to the data directory
- output_file: the name of the file in which to write the supertagged treebank; will be placed in the data folder
- split: optional, from "train", "test", "dev"; specify only for UD treebanks; only relevant for the `predict` mode; TODO: should also be relevant for the other modes to toggle UD imports/normal imports 
- standards_dir: path to the standards directory; default points to the pre-existing standards folder
- standard: the name of the standards file
- standard_from_xml: whether to ignore the `standard` argument and extract a standard from the stats.xml file of the UD treebank; works only for UD imports
- save_standard_from_xml: whether to save a standard extracted from a UD treebank; default true
- save_standard_from_xml_dir: where to save the standard extracted from a UD treebank; defaults to the pre-existing standards folder
- allow_partial_underspecification: TODO

#### Deprels settings

- arguments: list of dependency relations that should be treated as arguments
- adjuncts: list of dependency relations that should be treated as adjuncts
- delete: list of dependency relations that should be deleted; most likely only "root"
- labelled: whether to distinguish dependency relation types in the supertags
- subtypes: whether to distinguish subtypes (such as csubj:outer and csubj:pass) or ignore the part starting at the colon; TODO: currently the parser always ignores subtypes
- order_relations: whether to order arguments and head slots according to their surface order
- merged: optional; specify groups of dependency relations that should be merged to new supertypes
- merged_fallback_subtypes: whether to include dependency relations with a subtype whose full form does not occur in a merged group description in a merged group that refers to its main type (fallback)
- distinguish_merged_fallback_subtypes: whether to still retain the subtype in the `merged_fallback_subtypes` setting (true) or remove it in the case of a fallback (false)

**Attention:** all dependency relations specified in the standards file must be sorted into one of the three categories (arguments, adjuncts, delete).

#### Tagging settings

TODO

**Attention:** Currently, tagging settings should not be present when using the `extract` mode. TODO

## Modes

Run the program using the command `python -m deptag --settings <your_settings> <mode>`, where `<mode>` is one of the following:

### `vocab` Vocabulary creation

Creates a supertag vocabulary file that stores the supertag-to-id mapping extracted from the training data. Necessary to run before training.

### `extract` Supertag extraction

Extracts supertags from a conllu file, writes them to the misc field using the key `supertag=...` and reports statistics.

TODO: currently the method computes statistics for multiple treebanks and ignores the settings argument; add this as an explicit option instead.

### Parser training

### Parser evaluation

### Parser prediction

## General

### Supertag string representation

TODO