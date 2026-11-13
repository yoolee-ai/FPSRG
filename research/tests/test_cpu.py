"""CPU-only contracts. No model loading, generation, network or GPU calls."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Contracts(unittest.TestCase):
    def check_code(self, folder, code):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH'] = str(ROOT / folder) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        run = subprocess.run([sys.executable, '-c', code], cwd=ROOT / folder,
                             env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_source_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_off_portable_paths(self):
        self.check_code('off_engine', '''
import os
from pathlib import Path
os.environ.pop('FPSRG_HOME',None)
os.environ.pop('FPSRG_RESEARCH_CONFIG',None)
import common as c
root=Path.cwd().parents[1]
assert c.PACKAGE_ROOT==root
assert c.MODEL==root/'models/qwen35-base'
assert c.PARENT==root/'models/persononly-lora'
assert c.ROOT_RUN==root/'outputs/off'
assert c.RUN==root/'outputs/off/training'
''')

    def test_thinking_portable_paths(self):
        self.check_code('thinking_engine', '''
import os
from pathlib import Path
os.environ.pop('FPSRG_HOME',None)
os.environ.pop('FPSRG_RESEARCH_RUN',None)
import common as c
root=Path.cwd().parents[1]
assert c.PACKAGE_ROOT==root
assert c.MODEL==root/'models/huihui-six'
assert c.RUN==root/'outputs/thinking-refinement'
''')

    def test_lora_data_schedule(self):
        self.check_code('post_training', '''
from lora_objective import schedule, lr
rows=[{'sequence_tokens': (i % 41)+20} for i in range(1000)]
batches=schedule(rows,123)
assert len(batches)==125 and all(len(b)==8 for b in batches)
assert sorted(i for b in batches for i in b)==list(range(1000))
assert schedule(rows,123)==batches
''')

    def test_full_sft_denominators(self):
        self.check_code('post_training', '''
from full_sft_objective import objective_scale, rank_indices, learning_rate
assert objective_scale(True,400,8)==.02
assert objective_scale(False,400,8)==0.
assert rank_indices(list(range(16)),3)==(3,11)
assert abs(learning_rate(188)-5e-7)<1e-15
''')

    def test_off_targets(self):
        self.check_code('off_engine', '''
from historical_engine import shifted_target, neutral_alpha_for_step
import math
assert shifted_target(.5, 0, 1) == .5
assert abs(shifted_target(.5, 1, 1) - 1/(1+math.exp(-2))) < 1e-12
assert shifted_target(.5,-1,1) < .5 < shifted_target(.5,1,1)
assert neutral_alpha_for_step(1) is None
assert {neutral_alpha_for_step(i) for i in (0,4,8,12)} == {-1.,-.5,.5,1.}
''')

    def test_off_schedule(self):
        self.check_code('off_engine', '''
from ka_engine import step_spec
assert step_spec(0)['row_alphas'] == [.5,-.5]
assert step_spec(16)['row_alphas'] == [1.,-1.]
assert step_spec(239)['questionnaire_index'] == 29
try: step_spec(240)
except ValueError: pass
else: raise AssertionError('out-of-range training step accepted')
''')

    def test_off_probabilities(self):
        self.check_code('off_engine', '''
from engine import expected_score
assert expected_score([0,0,0,0,1], 'normal', '+') == 1
assert expected_score([0,0,0,0,1], 'reversed', '+') == 0
assert expected_score([0,0,0,0,1], 'normal', '-') == 0
assert expected_score([.2]*5, 'normal', '+') == .5
''')

    def test_complete_answer_scoring(self):
        self.check_code('thinking_engine', '''
from scoring import score, summary
case = dict(axis='A',kind='Q',targets={'high':'5'},mapping=0)
assert score(case,dict(complete=True,final='5')) == 100
assert score(case,dict(complete=False,final='5')) is None
assert score(case,dict(complete=True,final='5',thinking_closed=False)) is None
assert score(case,dict(complete=True,final='answer: 5')) is None
rows = [dict(case=case,result=dict(complete=True,final='5')),
        dict(case=case,result=dict(complete=False,final='5')),
        dict(case=dict(axis='A',kind='B',targets={'high':'B'},mapping=0),
             result=dict(complete=True,final='A'))]
x=summary(rows)
assert x['by_axis']['A']['S'] == 50
assert x['overall']['invalid_n'] == 1
assert x['by_axis']['A']['Q']['valid_n'] == 1
''')

    def test_thought_final_weights(self):
        self.check_code('thinking_engine', '''
from train_math import section_weights, candidates
w=section_weights(2,5,9)
assert w[:2] == [0.,0.]
assert abs(sum(w[2:5])-.7) < 1e-12
assert abs(sum(w[5:])-.3) < 1e-12
assert len(candidates()) == 48
from refinement import check_stop
check_stop()
''')

    def test_coverage_same_node(self):
        self.check_code('grid', '''
import numpy as np
from coverage import measure
scores=np.array([[50.]*5,[30.]*5,[70.]*5])
targets=np.array([[50.]*5,[80.]*5])
result, evidence=measure(scores,targets,np.array([True,False,False]),60)
assert result['raw']['covered']==2
assert result['gate98']['covered']==1
assert result['gate98']['coverage_percent']==50
assert (evidence['gate98_feasible_node']==[0,-1]).all()
''')

    def test_upstream_cache_identity(self):
        self.check_code('grid', '''
from joint_engine import AXES, prefix_key, grid_coefficients
from persistent_engine import retained_boundaries
layers=dict(zip(AXES,[40,32,32,16,32,32,40,32,40,32]))
a=[0]*10;b=list(a);b[0]=1
assert prefix_key(a,layers,40)==prefix_key(b,layers,40)
b[3]=1
assert prefix_key(a,layers,32)!=prefix_key(b,layers,32)
assert retained_boundaries((16,32,40))==(16,32)
try: grid_coefficients([2]*10,{'levels':[-1,0,1]})
except ValueError: pass
else: raise AssertionError('out-of-range grid coefficient accepted')
''')

    def test_digit_mapping(self):
        self.check_code('grid', '''
from measurement import score_digit
assert score_digit(1,'normal','+') == 0
assert score_digit(1,'reversed','+') == 100
assert score_digit(1,'reversed','-') == 0
assert score_digit(3,'normal','+') == 50
''')


if __name__ == '__main__':
    unittest.main(verbosity=2)
