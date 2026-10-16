def _deep_copy(value):
    if isinstance(value, dict):
        return {k: _deep_copy(v) for (k, v) in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple((_deep_copy(v) for v in value))
    if isinstance(value, set):
        return {_deep_copy(v) for v in value}
    return value

def predict_next_state(state, action):
    if not isinstance(state, dict):
        raise ValueError('state must be a dict.')
    if not isinstance(action, str):
        raise ValueError('action must be a string.')
    next_state = _deep_copy(state)
    if not isinstance(next_state, dict):
        return {}
    if 'step' in next_state and (not isinstance(next_state.get('step'), dict)):
        next_state['step'] = _deep_copy(state.get('step')) if isinstance(state.get('step'), dict) else {}
    return next_state
