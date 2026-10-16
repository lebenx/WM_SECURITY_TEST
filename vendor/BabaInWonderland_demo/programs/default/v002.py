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
    objects = next_state.get('objects')
    if not isinstance(objects, list):
        return next_state
    grid_size = next_state.get('grid_size')
    (dx, dy, facing) = _direction_delta(action)
    if facing is None:
        return next_state
    rules = _extract_rules(objects)
    you_words = set()
    for (subject, properties) in rules.items():
        if 'you' in properties:
            you_words.add(subject)
    if not you_words:
        return next_state
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        if obj.get('word') not in you_words:
            continue
        position = obj.get('position')
        if not isinstance(position, list) or len(position) != 2:
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        new_position = [x + dx, y + dy]
        obj['direction'] = facing
        if _in_bounds(new_position, grid_size):
            obj['position'] = new_position
    return next_state

def _direction_delta(action):
    if action == 'up':
        return (0, -1, 'facing up')
    if action == 'down':
        return (0, 1, 'facing down')
    if action == 'left':
        return (-1, 0, 'facing left')
    if action == 'right':
        return (1, 0, 'facing right')
    return (0, 0, None)

def _in_bounds(position, grid_size):
    if not isinstance(position, list) or len(position) != 2:
        return False
    if not isinstance(grid_size, list) or len(grid_size) != 2:
        return False
    x = position[0]
    y = position[1]
    width = grid_size[0]
    height = grid_size[1]
    if not isinstance(x, int) or not isinstance(y, int):
        return False
    if not isinstance(width, int) or not isinstance(height, int):
        return False
    return x >= 0 and y >= 0 and (x < width) and (y < height)

def _extract_rules(objects):
    text_at = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        position = obj.get('position')
        if not isinstance(position, list) or len(position) != 2:
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        text_at[x, y] = obj
    rules = {}
    directions = ((1, 0), (0, 1))
    for ((x, y), obj) in text_at.items():
        if obj.get('type') != 'rule_noun':
            continue
        subject = obj.get('word')
        if not isinstance(subject, str):
            continue
        for (dx, dy) in directions:
            is_obj = text_at.get((x + dx, y + dy))
            if not isinstance(is_obj, dict):
                continue
            if is_obj.get('type') != 'rule_operator' or is_obj.get('word') != 'is':
                continue
            cx = x + 2 * dx
            cy = y + 2 * dy
            expect_value = True
            while True:
                current = text_at.get((cx, cy))
                if not isinstance(current, dict):
                    break
                current_type = current.get('type')
                current_word = current.get('word')
                if expect_value:
                    if current_type == 'rule_property' and isinstance(current_word, str):
                        if subject not in rules:
                            rules[subject] = set()
                        rules[subject].add(current_word)
                        expect_value = False
                        cx += dx
                        cy += dy
                        continue
                    break
                if current_type == 'rule_operator' and current_word == 'and':
                    expect_value = True
                    cx += dx
                    cy += dy
                    continue
                break
    return rules
