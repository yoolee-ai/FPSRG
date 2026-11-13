"""Training candidates only; parent controller owns single-axis and grid stages."""
from common import AXES
TRAITS = AXES
LAYERS = (8, 16, 24, 32, 40, 48, 56, 60)
LAYER_QUEUE = (8, 60, 16, 56, 24, 48, 32, 40)
DEV_ALPHAS = (-1., -.5, 0., .5, 1.)
MAPPINGS = ('normal', 'reversed')


def candidates():
    return [dict(id=f'{axis}_L{layer:02d}_all', candidate_id=f'{axis}_L{layer:02d}_all',
                 trait=axis, feature=axis, layer=layer, train_mode='all')
            for axis in TRAITS for layer in LAYER_QUEUE]
