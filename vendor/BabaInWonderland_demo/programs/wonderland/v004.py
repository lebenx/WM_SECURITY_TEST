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
    objects = next_state.get('objects')
    grid_size = next_state.get('grid_size')
    if not isinstance(objects, list):
        return next_state
    rules = _extract_rules(objects)
    properties = rules.get('properties') if isinstance(rules, dict) else {}
    transforms = rules.get('transforms') if isinstance(rules, dict) else {}
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(transforms, dict):
        transforms = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        targets = transforms.get(word)
        if not isinstance(word, str) or not isinstance(targets, set):
            continue
        replacement = None
        for target in sorted(targets):
            if isinstance(target, str) and target != word:
                replacement = target
                break
        if replacement is not None:
            obj['word'] = replacement
    (dx, dy, facing) = _action_delta(action)
    if facing is None:
        return next_state
    if not (isinstance(grid_size, list) and len(grid_size) == 2 and isinstance(grid_size[0], int) and isinstance(grid_size[1], int)):
        return next_state
    width = grid_size[0]
    height = grid_size[1]
    if width <= 0 or height <= 0:
        return next_state
    controllable_words = set()
    for (word, props) in properties.items():
        if isinstance(props, set) and 'strange' in props:
            controllable_words.add(word)
    if not controllable_words:
        return next_state
    occupied = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2):
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        occupied.add((x, y))
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        if word not in controllable_words:
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2):
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        nx = x + dx
        ny = y + dy
        obj['direction'] = facing
        if nx < 0 or ny < 0 or nx >= width or (ny >= height):
            continue
        if (nx, ny) in occupied:
            continue
        obj['position'] = [nx, ny]
    return next_state

def _extract_rules(objects):
    rules = {'properties': {}, 'transforms': {}}
    if not isinstance(objects, list):
        return rules
    position_map = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2):
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        key = (x, y)
        if key not in position_map:
            position_map[key] = []
        position_map[key].append(obj)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'rule_noun':
            continue
        word = obj.get('word')
        position = obj.get('position')
        if not isinstance(word, str):
            continue
        if not (isinstance(position, list) and len(position) == 2):
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        for (dx, dy) in ((1, 0), (0, 1)):
            op_cell = position_map.get((x + dx, y + dy), [])
            target_cell = position_map.get((x + 2 * dx, y + 2 * dy), [])
            has_is = False
            for item in op_cell:
                if isinstance(item, dict) and item.get('type') == 'rule_operator' and (item.get('word') == 'is'):
                    has_is = True
                    break
            if not has_is:
                continue
            for item in target_cell:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'rule_property':
                    prop = item.get('word')
                    if isinstance(prop, str):
                        if word not in rules['properties']:
                            rules['properties'][word] = set()
                        rules['properties'][word].add(prop)
                elif item.get('type') == 'rule_noun':
                    target = item.get('word')
                    if isinstance(target, str):
                        if word not in rules['transforms']:
                            rules['transforms'][word] = set()
                        rules['transforms'][word].add(target)
    return rules

def _action_delta(action):
    if action == 'up':
        return (0, -1, 'facing up')
    if action == 'down':
        return (0, 1, 'facing down')
    if action == 'left':
        return (-1, 0, 'facing left')
    if action == 'right':
        return (1, 0, 'facing right')
    return (0, 0, None)
