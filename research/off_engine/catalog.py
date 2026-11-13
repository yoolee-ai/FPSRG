"""New Chinese research questions; not claimed validated scale translations."""
from collections import Counter
from common import ROOT_RUN, RUN, AXES, read

SOURCE_URLS = {}
ITEM_ALIASES = {}
TRANSLATION_STATUS = 'Original Chinese research items informed by classic theoretical definitions.'


def validate(rows):
    expected = Counter({(a, s): n for a in AXES for s, n in (('train', 30), ('dev', 6), ('test', 24))})
    if len(rows) != 300 or len({r['id'] for r in rows}) != 300:
        raise ValueError('expected_300_distinct_new_five_questionnaire_items')
    if Counter((r['trait'], r['split']) for r in rows) != expected:
        raise ValueError('new_questionnaire_split_counts')
    families = {}
    for row in rows:
        if row['feature'] != row['trait'] or row['axis'] != row['trait']:
            raise ValueError('questionnaire_axis_fields_disagree')
        if row['key'] not in ('+', '-') or row['polarity'] != (1 if row['key'] == '+' else -1):
            raise ValueError('invalid_questionnaire_polarity')
        if not isinstance(row['zh'], str) or not row['zh'].strip():
            raise ValueError('empty_questionnaire_text')
        key = (row['trait'], row['family_id'])
        if key in families and families[key] != row['split']:
            raise ValueError('questionnaire_family_crosses_split')
        families[key] = row['split']
    return rows


def items(split=None, feature=None):
    if split not in (None, 'train', 'dev', 'test') or feature not in (None, *AXES):
        raise ValueError('unknown_questionnaire_split_or_axis')
    path = RUN / 'data/catalog.json'
    if not path.exists():
        path = ROOT_RUN / 'data/new_questionnaire.json'
    rows = validate(read(path))
    # Legacy reporting carries an English-source field. These newly authored
    # Chinese questions have no English original, so keep that field empty.
    return [dict(row, en=row.get('en', '')) for row in rows
            if (split is None or row['split'] == split) and (feature is None or row['trait'] == feature)]

