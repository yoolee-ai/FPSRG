"""Exact five-axis questionnaire recipe, applied to the new R/T/V/M/S items."""
import legacy_protocol as p
from common import AXES

if tuple(AXES) not in (tuple('OCEAN'), tuple('RTVMS')):
    raise ValueError('Select the OCEAN or RTVMS five-axis refinement profile')
p.TRAITS = p.FEATURES = p.TRAIN_FEATURES = list(AXES)
p.FEATURE_NAMES.update(dict(R='风险偏好', T='未来后果考量', V='亲社会资源分配', M='物质价值取向', S='感官刺激敏感性'))
p.FEATURE_DEFINITIONS.update(dict(
    R='在风险边界明确且能承担损失的选择中，接受结果不确定性和波动的倾向',
    T='做当前决定时考虑较远期后果并愿意为未来结果调整当前选择的倾向',
    V='分配真实稀缺资源时，将他人的收益和共同利益纳入自己选择的倾向',
    M='将物质拥有、升级及其带来的幸福与成功评价放在生活中心的倾向',
    S='对声音、光线、气味、触感等感官刺激容易感到负担和过度刺激的倾向'))
p.ALPHAS = p.FINAL_ALPHAS = list(range(-2, 3))
p.TRAIN_MODES = p.MODES = ['all']
p.TEST_REPEATS = p.CONTROL_REPEATS = 1
p.CANDIDATES = p.candidates()
from legacy_protocol import *
