"""CPU-only protocol tests; transport is mocked, never a model or external API."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx
from fastapi.testclient import TestClient

from .app import (AXES, build_messages, context_preview, create_app,
                  emotion_control, match_levels, normalize_input, semantic_control)
from .config import Settings
from .definitions import DEFINITIONS


CATALOG = dict(artifact_id='synthetic-test-six', axes=list(AXES),
    generation={'thinking': True}, emotion={'joy': 2, 'anger': 2}, nodes=[
        dict(node_id=10, persona_id=1, scores={axis: 50. for axis in AXES},
             coefficients={axis: 0. for axis in AXES}),
        dict(node_id=20, persona_id=2, scores={axis: 75. for axis in AXES},
             coefficients={axis: 1. for axis in AXES})])


def input_data():
    return {'people': [{'id': 'synthetic', 'name': 'Synthetic',
            'levels': {axis: 'medium' for axis in AXES},
            'dialogue': [{'role': 'customer', 'text': 'This is synthetic source text.'}]}]}


class UnitTests(unittest.TestCase):
    def test_full_extraction_rubric(self):
        self.assertEqual(set(DEFINITIONS), set(AXES))

    def test_level_matching(self):
        found = match_levels({axis: 'high' for axis in AXES}, CATALOG)
        self.assertEqual(found['node_id'], 20)
        self.assertEqual(found['level_mismatches'], 0)
        self.assertEqual(found['mean_abs_error'], 0)

    def test_unknown_dimension_rejected(self):
        with self.assertRaises(ValueError):
            match_levels({'A': 'low'}, CATALOG)

    def test_emotion_fixed_two(self):
        self.assertEqual(emotion_control('joy')['strength'], 2)
        self.assertEqual(emotion_control('anger')['strength'], 2)
        self.assertEqual(emotion_control('none')['strength'], 0)
        with self.assertRaises(ValueError):
            emotion_control('joy3')

    def test_no_context_and_no_history_thought(self):
        person = dict(levels={axis: 'medium' for axis in AXES}, matched=CATALOG['nodes'][0],
            shared_history=[{'role': 'assistant', 'content': 'Prior answer', 'reasoning': 'Private old reasoning'}],
            context='Personal source must not leak')
        context = context_preview(person, 'none')
        self.assertEqual(context['history'], [])
        self.assertEqual(context['prefix_messages'], [])
        messages, _ = build_messages(person, 'Now', 'none', False)
        self.assertEqual(len(messages), 2)
        self.assertNotIn('Prior answer', json.dumps(messages))
        messages, _ = build_messages(person, 'Now', 'shared', False)
        serialized = json.dumps(messages)
        self.assertIn('Prior answer', serialized)
        self.assertNotIn('Private old reasoning', serialized)
        self.assertNotIn('Personal source must not leak', serialized)

    def test_support_preserves_scores(self):
        person = dict(levels={axis: 'high' for axis in AXES}, matched=copy.deepcopy(CATALOG['nodes'][0]))
        before = copy.deepcopy(person)
        result = semantic_control(person)
        self.assertTrue(result['semantic_supplement'])
        self.assertFalse(result['measured_after_supplement'])
        self.assertEqual(person, before)
        self.assertFalse(semantic_control(person, False)['semantic_supplement'])

    def test_invalid_manual_levels(self):
        data = input_data()
        data['people'][0]['levels']['A'] = 'extreme'
        with self.assertRaises(ValueError):
            normalize_input(json.dumps(data))


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calls = []
        self.bad_emotion = False
        self.finish_reason = 'stop'
        settings = Settings(Path(self.temp.name), 'http://vector.test', 'synthetic-test-token',
                            'https://extractor.test', '', 'test-model', True, 100, 5)
        def handle(request):
            self.calls.append(request)
            if request.url.path == '/catalog':
                return httpx.Response(200, json=CATALOG)
            if request.url.path == '/health':
                return httpx.Response(200, json={'ready': True})
            body = json.loads(request.content)
            self.assertEqual(request.headers['authorization'], 'Bearer synthetic-test-token')
            self.assertEqual(body['node_id'], 10)
            emotion = 'none' if self.bad_emotion else body['emotion']
            return httpx.Response(200, json={
                'choices': [{'finish_reason': self.finish_reason, 'message': {
                    'content': 'Synthetic answer', 'reasoning_content': 'Synthetic reasoning'}}],
                'grid': CATALOG['nodes'][0],
                'metadata': {'artifact_id': CATALOG['artifact_id'], 'emotion': emotion,
                             'emotion_strength': 0 if emotion == 'none' else 2,
                             'thinking_closed': self.finish_reason == 'stop', 'runtime_persona_id': 730,
                             'complete': self.finish_reason == 'stop', 'ended_with_eos': self.finish_reason == 'stop'},
                'usage': {'completion_tokens': 8}})
        self.app = create_app(settings, httpx.MockTransport(handle))
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.client.get('/')

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def wait_job(self, ticket):
        for _ in range(100):
            record = self.client.get(ticket['status_url']).json()
            if record['state'] != 'running':
                return record
            time.sleep(.005)
        self.fail('mock job did not terminate')

    def make_person(self):
        response = self.client.post('/api/extract', json={'text': json.dumps(input_data())})
        self.assertEqual(response.status_code, 200, response.text)
        result = self.wait_job(response.json())
        self.assertEqual(result['state'], 'complete', result)
        return result['result']['individuals'][0]

    def test_manual_extraction_no_external_calls(self):
        person = self.make_person()
        self.assertEqual(person['matched']['node_id'], 10)
        self.assertTrue(all(request.url.host == 'vector.test' for request in self.calls))
        self.assertEqual(self.client.get('/api/status').json()['catalog_size'], 2)

    def test_external_extraction_requires_consent_and_key(self):
        data = input_data()
        del data['people'][0]['levels']
        self.assertEqual(self.client.post('/api/extract', json={'text': json.dumps(data)}).status_code, 400)
        data['allow_external_extraction'] = True
        self.assertEqual(self.client.post('/api/extract', json={'text': json.dumps(data)}).status_code, 503)
        self.assertEqual(len(self.calls), 0)

    def test_emotion_receipt_and_history(self):
        person = self.make_person()
        response = self.client.post('/api/chat', json={'individual_id': person['id'],
                                   'question': 'Hello', 'context_mode': 'shared', 'emotion': 'joy'})
        result = self.wait_job(response.json())
        self.assertTrue(result['result']['complete'])
        self.assertEqual(result['result']['emotion']['strength'], 2)
        stored = self.client.get('/api/people/' + person['id']).json()
        self.assertEqual(len(stored['shared_history']), 2)
        response = self.client.post('/api/chat', json={'individual_id': person['id'],
                                   'question': 'Again', 'context_mode': 'none', 'emotion': 'none'})
        self.wait_job(response.json())
        last = json.loads([call for call in self.calls if call.url.path.endswith('/completions')][-1].content)
        self.assertEqual(len(last['messages']), 2)
        self.assertEqual(last['emotion'], 'none')

    def test_wrong_emotion_rejected(self):
        person = self.make_person()
        self.bad_emotion = True
        ticket = self.client.post('/api/chat', json={'individual_id': person['id'], 'question': 'Hi', 'emotion': 'anger'}).json()
        result = self.wait_job(ticket)
        self.assertEqual(result['state'], 'error')
        stored = self.client.get('/api/people/' + person['id']).json()
        self.assertEqual(stored['shared_history'], [])
        self.assertIsNone(stored['pending'])

    def test_incomplete_not_added_to_context(self):
        person = self.make_person()
        self.finish_reason = 'length'
        ticket = self.client.post('/api/chat', json={'individual_id': person['id'], 'question': 'Hi'}).json()
        result = self.wait_job(ticket)
        self.assertFalse(result['result']['complete'])
        self.assertEqual(result['result']['answer'], 'Synthetic answer')
        self.assertEqual(self.client.get('/api/people/' + person['id']).json()['shared_history'], [])

    def test_browser_session_and_csrf_isolation(self):
        person = self.make_person()
        original = self.client.cookies.get('fpsrg_session')
        self.client.cookies.clear()
        self.assertEqual(self.client.get('/api/people/' + person['id']).status_code, 404)
        self.client.cookies.set('fpsrg_session', original)
        response = self.client.post('/api/extract', json={'text': json.dumps(input_data())},
                                    headers={'origin': 'https://other.example'})
        self.assertEqual(response.status_code, 403)

    def test_history_reset_mode_isolated(self):
        person = self.make_person()
        ticket = self.client.post('/api/chat', json={'individual_id': person['id'], 'question': 'Hi'}).json()
        self.wait_job(ticket)
        self.client.delete('/api/people/' + person['id'] + '/history?mode=none')
        self.assertEqual(len(self.client.get('/api/people/' + person['id']).json()['shared_history']), 2)

    def test_six_external_axes_mocked_and_audited(self):
        settings = Settings(Path(self.temp.name) / 'external', 'http://vector.test', '',
                            'https://extractor.test', 'test-only-key', 'test-model', True, 100, 5)
        calls = []
        def handle(request):
            if request.url.path == '/catalog':
                return httpx.Response(200, json=CATALOG)
            payload = json.loads(request.content)
            data = json.loads(payload['messages'][1]['content'])
            calls.append(data['axis'])
            return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
                'content': json.dumps({'people': [{'id': person['id'], 'level': 'medium',
                    'confidence': 'low', 'evidence': 'Synthetic test evidence'} for person in data['people']]})}}]})
        isolated_app = create_app(settings, httpx.MockTransport(handle))
        with TestClient(isolated_app) as client:
            client.get('/')
            data = input_data()
            del data['people'][0]['levels']
            data['allow_external_extraction'] = True
            ticket = client.post('/api/extract', json={'text': json.dumps(data)}).json()
            for _ in range(100):
                result = client.get(ticket['status_url']).json()
                if result['state'] != 'running':
                    break
                time.sleep(.005)
            self.assertEqual(result['state'], 'complete', result)
            self.assertEqual(set(calls), set(AXES))
            self.assertEqual(len(calls), 6)
        files = list((settings.home / 'runtime' / 'personlab').rglob('*.json'))
        self.assertGreater(len(files), 0)
        self.assertFalse(any('test-only-key' in file.read_text() for file in files))


if __name__ == '__main__':
    unittest.main()
