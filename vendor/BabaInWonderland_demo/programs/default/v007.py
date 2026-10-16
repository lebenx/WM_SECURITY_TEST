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
    rules = _extract_rules(objects)
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        if not isinstance(word, str):
            continue
        subject_rules = rules.get(word)
        if not isinstance(subject_rules, dict):
            continue
        noun_targets = subject_rules.get('nouns')
        if not isinstance(noun_targets, list) or len(noun_targets) != 1:
            continue
        target_word = noun_targets[0]
        if isinstance(target_word, str):
            obj['word'] = target_word
    rules = _extract_rules(objects)
    you_words = set()
    stop_words = set()
    push_words = set()
    move_words = set()
    for (subject, subject_rules) in rules.items():
        if not isinstance(subject_rules, dict):
            continue
        properties = subject_rules.get('properties')
        if not isinstance(properties, set):
            continue
        if 'you' in properties:
            you_words.add(subject)
        if 'stop' in properties:
            stop_words.add(subject)
        if 'push' in properties:
            push_words.add(subject)
        if 'move' in properties:
            move_words.add(subject)
    grid_size = next_state.get('grid_size')
    position_objects = _collect_position_objects(objects)
    (dx, dy, facing) = _direction_delta(action)
    if facing is not None:
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
            obj['direction'] = facing
            _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words)
    _apply_move_properties(objects, grid_size, position_objects, push_words, stop_words, move_words)
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
                            rules[subject] = {'properties': set(), 'nouns': []}
                        rules[subject]['properties'].add(current_word)
                        expect_value = False
                        cx += dx
                        cy += dy
                        continue
                    if current_type == 'rule_noun' and isinstance(current_word, str):
                        if subject not in rules:
                            rules[subject] = {'properties': set(), 'nouns': []}
                        if current_word not in rules[subject]['nouns']:
                            rules[subject]['nouns'].append(current_word)
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

def _blocked_by_stop(position, occupied_positions, stop_words):
    if not isinstance(position, list) or len(position) != 2:
        return True
    x = position[0]
    y = position[1]
    if not isinstance(x, int) or not isinstance(y, int):
        return True
    words = occupied_positions.get((x, y))
    if not isinstance(words, list):
        return False
    for word in words:
        if word in stop_words:
            return True
    return False

def _collect_position_objects(objects):
    position_objects = {}
    if not isinstance(objects, list):
        return position_objects
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
        key = (x, y)
        if key not in position_objects:
            position_objects[key] = []
        position_objects[key].append(obj)
    return position_objects

def _is_pushable(obj, push_words):
    if not isinstance(obj, dict):
        return False
    obj_type = obj.get('type')
    if obj_type != 'world_object':
        return True
    word = obj.get('word')
    return isinstance(word, str) and word in push_words

def _is_stopping(obj, push_words, stop_words):
    if not isinstance(obj, dict):
        return False
    if _is_pushable(obj, push_words):
        return False
    if obj.get('type') != 'world_object':
        return False
    word = obj.get('word')
    return isinstance(word, str) and word in stop_words

def _relocate_object(obj, new_position, position_objects):
    if not isinstance(obj, dict):
        return
    old_position = obj.get('position')
    if isinstance(old_position, list) and len(old_position) == 2:
        old_x = old_position[0]
        old_y = old_position[1]
        if isinstance(old_x, int) and isinstance(old_y, int):
            old_key = (old_x, old_y)
            old_list = position_objects.get(old_key)
            if isinstance(old_list, list):
                kept = []
                for item in old_list:
                    if item is not obj:
                        kept.append(item)
                if kept:
                    position_objects[old_key] = kept
                elif old_key in position_objects:
                    del position_objects[old_key]
    obj['position'] = [new_position[0], new_position[1]]
    new_key = (new_position[0], new_position[1])
    if new_key not in position_objects:
        position_objects[new_key] = []
    position_objects[new_key].append(obj)

def _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words):
    if not isinstance(obj, dict):
        return False
    position = obj.get('position')
    if not isinstance(position, list) or len(position) != 2:
        return False
    x = position[0]
    y = position[1]
    if not isinstance(x, int) or not isinstance(y, int):
        return False
    target = [x + dx, y + dy]
    if not _in_bounds(target, grid_size):
        return False
    blockers = position_objects.get((target[0], target[1]), [])
    if not isinstance(blockers, list):
        blockers = []
    blockers = list(blockers)
    pushables = []
    for blocker in blockers:
        if blocker is obj:
            continue
        if _is_pushable(blocker, push_words):
            pushables.append(blocker)
            continue
        if _is_stopping(blocker, push_words, stop_words):
            return False
    for blocker in pushables:
        if not _attempt_move(blocker, dx, dy, grid_size, position_objects, push_words, stop_words):
            return False
    _relocate_object(obj, target, position_objects)
    facing = _delta_to_facing(dx, dy)
    if facing is not None:
        obj['direction'] = facing
    return True

def _facing_to_delta(facing):
    if facing == 'facing up':
        return (0, -1)
    if facing == 'facing down':
        return (0, 1)
    if facing == 'facing left':
        return (-1, 0)
    if facing == 'facing right':
        return (1, 0)
    return (0, 0)

def _opposite_facing(facing):
    if facing == 'facing up':
        return 'facing down'
    if facing == 'facing down':
        return 'facing up'
    if facing == 'facing left':
        return 'facing right'
    if facing == 'facing right':
        return 'facing left'
    return facing

def _apply_move_properties(objects, grid_size, position_objects, push_words, stop_words, move_words):
    if not isinstance(objects, list):
        return
    movers = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        if not isinstance(word, str) or word not in move_words:
            continue
        movers.append(obj)
    for obj in movers:
        facing = obj.get('direction')
        if not isinstance(facing, str):
            facing = 'facing right'
            obj['direction'] = facing
        (dx, dy) = _facing_to_delta(facing)
        moved = False
        if dx != 0 or dy != 0:
            moved = _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words)
        if moved:
            continue
        new_facing = _opposite_facing(facing)
        obj['direction'] = new_facing
        (dx, dy) = _facing_to_delta(new_facing)
        if dx == 0 and dy == 0:
            continue
        _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words)

def _delta_to_facing(dx, dy):
    if dx == 0 and dy == -1:
        return 'facing up'
    if dx == 0 and dy == 1:
        return 'facing down'
    if dx == -1 and dy == 0:
        return 'facing left'
    if dx == 1 and dy == 0:
        return 'facing right'
    return None
