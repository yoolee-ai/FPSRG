"""One CPU-only prompt contract for behavior training, selection and generation."""
import hashlib
import json

SYSTEM = '你是一个人，根据场景事实回答。'
RESPONSE_INSTRUCTION = '结合当前场景和对方的话，以第一人称直接呈现你此刻会说的话或会采取的行动，也可以同时包含两者。'
CHANNELS = dict(face_to_face='当面交谈', text='文字私聊', voice='语音对话', group_chat='群聊')
INPUT_FIELDS = ('scene', 'interlocutor', 'interaction_channel', 'last_utterance', 'response_instruction')


def render_behavior_input(row):
    for key in INPUT_FIELDS:
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'missing_or_empty_input_field:{key}')
    if row['response_instruction'] != RESPONSE_INSTRUCTION:
        raise ValueError('noncanonical_response_instruction')
    if row['interaction_channel'] not in CHANNELS:
        raise ValueError('unknown_interaction_channel')
    return (f'【场景事实】\n{row["scene"]}\n\n'
            f'【正在与你交谈的人】\n{row["interlocutor"]}\n\n'
            f'【交流方式】\n{CHANNELS[row["interaction_channel"]]}\n\n'
            f'【对方刚刚说】\n{row["last_utterance"]}\n\n'
            f'{RESPONSE_INSTRUCTION}')


def input_messages(row):
    return [dict(role='system', content=SYSTEM),
            dict(role='user', content=render_behavior_input(row))]


def prefix_sha256(row):
    payload = json.dumps(input_messages(row), ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def training_branch(row, side):
    """The sign and target are metadata, never appended to the user prompt."""
    if side not in ('high', 'low'):
        raise ValueError('unknown_target_side')
    return dict(input_messages=input_messages(row), input_sha256=prefix_sha256(row),
                target_response=row[f'{side}_response'], injection_sign=1 if side == 'high' else -1)

