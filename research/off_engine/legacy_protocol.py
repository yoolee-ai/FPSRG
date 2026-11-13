"""Predeclared five-vector experiment using disjoint IPIP-NEO source items.

The car engine's premium field is a compatibility sentinel, never an economic
quantity here. Every positive coefficient targets a HIGH O/C/E/A/N score. High
N means neuroticism, not emotional stability. Synthetic high/low endpoints are
constructed from published scoring keys, not observed human questionnaire data.
"""
from collections import Counter
from copy import deepcopy
import re

try:
    from . import catalog
except ImportError:
    import catalog

TRAITS = ['O', 'C', 'E', 'A', 'N']
FEATURES = TRAITS
TRAIN_FEATURES = TRAITS
FEATURE_NAMES = {'O': '开放性', 'C': '尽责性', 'E': '外向性',
                 'A': '宜人性', 'N': '神经质'}
FEATURE_DEFINITIONS = {
    'O': '对想象、艺术、情感体验、新体验、思考和不同观念的开放程度',
    'C': '做事的自律、条理、责任感、目标坚持和审慎程度',
    'E': '社交接近、群体互动、主动表达、活跃和积极情绪的倾向',
    'A': '信任、坦诚、关心他人、合作、谦逊和同情的倾向',
    'N': '容易体验焦虑、愤怒、低落、不自在、失控和压力脆弱性的倾向',
}
LAYERS = [8, 16, 24, 32, 40, 48, 56, 60]
TRAIN_MODES = ['prefill', 'all']
MODES = TRAIN_MODES
EVAL_MODE = 'all'
TRAIN_ALPHAS = [-1.0, -0.5, 0.5, 1.0]
DEV_ALPHAS = [-1.0, -0.5, 0.0, 0.5, 1.0]
ALPHAS = [i / 5 for i in range(-10, 11)]
FINAL_ALPHAS = ALPHAS
COARSE = DEV_ALPHAS
TRAIN_PREMIUMS = [0.0]
DEV_PREMIUMS = [0.0]
TEST_PREMIUMS = [0.0]
TEST_REPEATS = 20
CONTROL_REPEATS = 10
MAPPINGS = ('normal', 'reversed')
ANCHORS = {1: '非常不符合', 2: '比较不符合', 3: '既不符合也不否认',
           4: '比较符合', 5: '非常符合'}

# Explicit copy of experiments/attribute_wtp/protocol.py NEUTRAL, frozen here.
NEUTRAL = [
    {'id': 'N01', 'prompt': '七加五等于多少？请只写结果。', 'response': '12'},
    {'id': 'N02', 'prompt': '九减四等于多少？请只写结果。', 'response': '5'},
    {'id': 'N03', 'prompt': '三乘六等于多少？请只写结果。', 'response': '18'},
    {'id': 'N04', 'prompt': '二十四除以六等于多少？请只写结果。', 'response': '4'},
    {'id': 'N05', 'prompt': '一小时有多少分钟？请只写数字。', 'response': '60'},
    {'id': 'N06', 'prompt': '一年有多少个月？请只写数字。', 'response': '12'},
    {'id': 'N07', 'prompt': '水的化学式是什么？请只写化学式。', 'response': 'H2O'},
    {'id': 'N08', 'prompt': '三角形有几条边？请只写数字。', 'response': '3'},
]


def _feature(feature, allow_none=False):
    if feature not in TRAITS and not (allow_none and feature is None):
        raise ValueError(f'Unknown feature: {feature!r}')


def _repeats(repeats):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')


def role_prefix(role=None, feature='R'):
    _feature(feature)
    if role not in (None, 'bare', 'low', 'high'):
        raise ValueError(f'Unknown role: {role!r}')
    if role in (None, 'bare'):
        return ''
    level = '较高' if role == 'high' else '较低'
    return (f'请按以下指定人格取向回答：我的{FEATURE_NAMES[feature]}倾向{level}。'
            f'这里的{FEATURE_NAMES[feature]}指{FEATURE_DEFINITIONS[feature]}。'
            '此设定只指定这一维度，不另外指定其他人格维度。'
            '请以这一取向下自己的通常情况作答，不要介绍两类人的区别。\n\n')


def scenarios(split='train', feature=None):
    _feature(feature, allow_none=True)
    return [dict(row, benefit_units=1, gain_positive=True,
                 context='请根据自己目前通常的情况回答，而不是理想中的自己。',
                 feature_name=FEATURE_NAMES[row['feature']])
            for row in catalog.items(split, feature)]


def _options(s, mapping):
    if mapping not in ('A', 'B'):
        raise ValueError(f'Unknown mapping: {mapping!r}')
    if s['key'] not in ('+', '-'):
        raise ValueError('Unknown scoring key')
    agree, disagree = '这项描述很符合我。', '这项描述很不符合我。'
    high, low = (agree, disagree) if s['key'] == '+' else (disagree, agree)
    return (high, low) if mapping == 'A' else (low, high)


def choice_prompt(s, premium=0, mapping='A', role=None):
    a, b = _options(s, mapping)
    return (role_prefix(role, s['feature']) + s['context']
            + f'\n描述：{s["zh"]}\nA：{a}\nB：{b}\n'
            + '哪一项更贴近我的通常倾向？先单独一行写 A 或 B，'
            + '再用一句简短中文说明这项描述是否符合自己的日常倾向。必须选择一项。')


def answer_pair(s, premium=0, mapping='A'):
    """High-trait response then low-trait response, with reverse-key handling."""
    _options(s, mapping)
    if s['split'] != 'train':
        raise ValueError('Endpoint training targets may only use train items')
    def reason(agreement):
        relation = '很符合' if agreement else '很不符合'
        return f'这项描述“{s["zh"]}”{relation}我的日常倾向，所以我选择相应的一项。'
    other = 'B' if mapping == 'A' else 'A'
    return (f'{mapping}\n{reason(s["key"] == "+")}',
            f'{other}\n{reason(s["key"] == "-")}')


def jobs(split, feature=None, premiums=None, repeats=1, role=None):
    """A/B rows for the historical training engine; premiums are ignored."""
    _repeats(repeats)
    rows = []
    for s in scenarios(split, feature):
        for mapping in ('A', 'B'):
            for repeat in range(repeats):
                ident = f'choice:{split}:{s["id"]}:{mapping}:r{repeat:04d}'
                rows.append(dict(
                    id=ident, job_id=ident, scenario_id=s['id'], item_id=s['id'],
                    feature=s['feature'], trait=s['trait'], facet=s['facet'],
                    family_id=s['family_id'], split=split, key=s['key'],
                    polarity=s['polarity'], premium=0.0, mapping=mapping,
                    high_trait=mapping, repeat=repeat, replica=repeat,
                    role=None if role == 'bare' else role,
                    prompt=choice_prompt(s, 0.0, mapping, role),
                ))
    return rows


def train_jobs(feature=None):
    return jobs('train', feature=feature)


def likert_prompt(item, mapping='normal', role=None):
    if mapping not in MAPPINGS:
        raise ValueError(f'Unknown mapping: {mapping!r}')
    meanings = [(label, label if mapping == 'normal' else 6 - label)
                for label in range(1, 6)]
    options = '\n'.join(f'{label}：{ANCHORS[rating]}' for label, rating in meanings)
    return (role_prefix(role, item['feature'])
            + '请根据自己目前通常的情况回答，而不是理想中的自己。\n'
            + f'描述：{item["zh"]}\n这项描述在多大程度上符合我？\n{options}\n'
            + '先单独一行写一个选项数字（1、2、3、4或5），'
            + '再用一句简短中文说明理由。不要列出其他选项。')


def likert_jobs(split, feature=None, repeats=1, role=None, namespace=None):
    _feature(feature, allow_none=True)
    _repeats(repeats)
    if role not in (None, 'bare', 'high', 'low'):
        raise ValueError(f'Unknown role: {role!r}')
    if namespace is None:
        namespace = (f'{split}_likert' if role is None
                     else f'{split}_controls_{role}')
    if not isinstance(namespace, str) or not re.fullmatch(r'[A-Za-z0-9_]+', namespace):
        raise ValueError('namespace must be an ASCII identifier')
    rows = []
    for item in catalog.items(split, feature):
        for mapping in MAPPINGS:
            for repeat in range(repeats):
                ident = f'{namespace}:{item["id"]}:{mapping}:r{repeat:04d}'
                rows.append(dict(
                    id=ident, job_id=ident, scenario_id=item['id'], item_id=item['id'],
                    feature=item['feature'], trait=item['trait'], facet=item['facet'],
                    family_id=item['family_id'], key=item['key'], polarity=item['polarity'],
                    split=split, raw_item=item['en'], en=item['en'], zh=item['zh'],
                    mapping=mapping, repeat=repeat, replica=repeat,
                    role=None if role == 'bare' else role, namespace=namespace,
                    prompt=likert_prompt(item, mapping, role),
                ))
    return rows


def parse_likert(text, row):
    """Require one digit on its own first line followed by one Chinese reason.

    This checks output structure, not semantic consistency or EOS. The engine
    separately records generation/EOS validity. Invalid output stays unscored.
    """
    if row['mapping'] not in MAPPINGS or row['key'] not in ('+', '-'):
        raise ValueError('Unknown mapping or scoring key')
    stripped = text.strip() if isinstance(text, str) else ''
    first, _, tail = stripped.partition('\n')
    reason = tail.strip()
    reason_present = bool(reason and re.search(r'[\u3400-\u9fff]', reason))
    digit = int(first) if re.fullmatch(r'[1-5]', first) else None
    format_valid = (digit is not None and reason_present
                    and '\n' not in reason and '\r' not in reason)
    if format_valid:
        rating = digit if row['mapping'] == 'normal' else 6 - digit
        trait_score = rating if row['key'] == '+' else 6 - rating
    else:
        rating = trait_score = None
    return dict(
        raw_label=digit, selected_digit=digit, rating=rating,
        trait_score=trait_score,
        score01=(trait_score - 1) / 4 if trait_score is not None else None,
        format_valid=bool(format_valid), valid=bool(format_valid),
        reason_present=reason_present, reason=reason,
        invalid_reason=(None if format_valid else 'missing_or_invalid_digit'
                        if digit is None else 'missing_or_invalid_reason'),
    )


def candidates():
    return [dict(id=f'{feature}_L{layer:02d}_{mode}',
                 candidate_id=f'{feature}_L{layer:02d}_{mode}',
                 feature=feature, trait=feature, layer=layer, train_mode=mode)
            for feature in TRAITS for layer in LAYERS for mode in TRAIN_MODES]


CANDIDATES = candidates()


def final_conditions(feature=None):
    _feature(feature, allow_none=True)
    rows = []
    for trait in (TRAITS if feature is None else [feature]):
        for alpha in ALPHAS:
            ident = f'{trait}_vector_{alpha:+.1f}'
            rows.append(dict(
                id=ident, condition_id=ident, feature=trait, trait=trait,
                method='vector', alpha=alpha, role=None, rows_expected=960,
                input_data=f'data/{trait}/test_likert.json',
                repeats=TEST_REPEATS, eval_mode=EVAL_MODE,
                namespace='test_likert',
            ))
        for role in ('bare', 'high', 'low'):
            ident = f'{trait}_control_{role}'
            rows.append(dict(
                id=ident, condition_id=ident, feature=trait, trait=trait,
                method='role_control', alpha=None, role=None if role == 'bare' else role,
                rows_expected=480, input_data=f'data/{trait}/test_controls_{role}.json',
                repeats=CONTROL_REPEATS, eval_mode=EVAL_MODE,
                namespace=f'test_controls_{role}',
            ))
    return rows


def manifest():
    banks = {trait: {split: catalog.items(split, trait)
                     for split in ('train', 'dev', 'test')} for trait in TRAITS}
    pairs = []
    for trait in TRAITS:
        lookup = {s['id']: s for s in scenarios('train', trait)}
        for row in jobs('train', feature=trait):
            high, low = answer_pair(lookup[row['item_id']], 0.0, row['mapping'])
            pairs.append(dict(row, high_response=high, low_response=low))
    return dict(
        schema_version=1, experiment='bigfive_standard_scale_vectors_20260916',
        traits=list(TRAITS), features=dict(FEATURE_NAMES),
        positive_direction='higher O/C/E/A/N; high N is neuroticism',
        sources=dict(catalog.SOURCE_URLS), aliases=dict(catalog.ITEM_ALIASES),
        translation_status=catalog.TRANSLATION_STATUS,
        final_scale='Johnson (2014) IPIP-NEO-120, all 120 items, research Chinese translation',
        split_counts={'train': 150, 'dev': 30, 'test': 120},
        split_counts_per_feature={'train': 30, 'dev': 6, 'test': 24},
        dev_key_counts={trait: dict(Counter(row['key'] for row in banks[trait]['dev']))
                        for trait in TRAITS},
        item_banks=banks, train_answer_pairs=pairs, neutral=deepcopy(NEUTRAL),
        layers=list(LAYERS), train_modes=list(TRAIN_MODES), eval_mode=EVAL_MODE,
        train_alphas=list(TRAIN_ALPHAS), dev_alphas=list(DEV_ALPHAS),
        final_alphas=list(ALPHAS), candidates=candidates(),
        final_conditions=final_conditions(),
        invariants={
            'final_items_not_used_for_training_initialization_or_selection': True,
            'synthetic_endpoints_are_not_human_answers': True,
            'semantic_novelty_or_pretraining_exclusion_claimed': False,
            'dev_split_selected_without_model_outputs': True,
            'existing_ipip50_modified': False,
            'test_mapping_pooling': 'equal-weight semantic scores; retain both mappings',
            'baseline_namespace': '20-repeat vector alpha zero differs from 10-repeat bare control',
        },
    )

