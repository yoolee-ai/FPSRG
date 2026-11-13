"""Pure CPU reference for post-scheduler-reordering request-to-token routing."""
import numpy as np


def token_routes(req_ids, scheduled, personas, num_input_tokens, total_tokens, capacity, out=None):
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Invalid bank capacity')
    if type(num_input_tokens) is not int or type(total_tokens) is not int or not 0 <= total_tokens <= num_input_tokens:
        raise ValueError('Invalid token counts')
    if len(req_ids) != len(set(req_ids)):
        raise ValueError('Duplicate request ID')
    if out is None:
        out=np.zeros(num_input_tokens, dtype=np.int64)
    if out.dtype != np.int64 or out.ndim != 1 or out.size < num_input_tokens:
        raise ValueError('Invalid routing buffer')
    out[:num_input_tokens]=0; offset=0
    for rid in req_ids:
        count=scheduled.get(rid,0)
        if type(count) is not int or count < 0:
            raise ValueError('Invalid scheduled count')
        if not count: continue
        row=personas.get(rid)
        if type(row) is not int or not 0 <= row < capacity or offset+count>num_input_tokens:
            raise ValueError('Invalid persona or routing overflow')
        out[offset:offset+count]=row; offset+=count
    if offset != total_tokens or sum(scheduled.values()) != offset:
        raise ValueError('Scheduled-token routing mismatch')
    return out[:num_input_tokens]
