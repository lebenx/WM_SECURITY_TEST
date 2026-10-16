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
    _apply_noun_transformations(objects, rules)
    rules = _extract_rules(objects)
    property_sets = _extract_property_word_sets(rules)
    you_words = property_sets.get('you', set())
    stop_words = property_sets.get('stop', set())
    push_words = property_sets.get('push', set())
    move_words = property_sets.get('move', set())
    open_words = property_sets.get('open', set())
    shut_words = property_sets.get('shut', set())
    float_words = property_sets.get('float', set())
    grid_size = next_state.get('grid_size')
    position_objects = _collect_position_objects(objects)
    (dx, dy, facing) = _direction_delta(action)
    if facing is not None:
        you_objects = []
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
            you_objects.append(obj)
        for obj in _ordered_movers(you_objects, dx, dy):
            _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words, open_words, shut_words)
    _apply_move_properties(objects, grid_size, position_objects, push_words, stop_words, move_words, open_words, shut_words)
    rules = _extract_rules(objects)
    _apply_noun_transformations(objects, rules)
    rules = _extract_rules(objects)
    property_sets = _extract_property_word_sets(rules)
    you_words = property_sets.get('you', set())
    stop_words = property_sets.get('stop', set())
    push_words = property_sets.get('push', set())
    open_words = property_sets.get('open', set())
    shut_words = property_sets.get('shut', set())
    sink_words = property_sets.get('sink', set())
    defeat_words = property_sets.get('defeat', set())
    float_words = property_sets.get('float', set())
    win_words = property_sets.get('win', set())
    hot_words = property_sets.get('hot', set())
    melt_words = property_sets.get('melt', set())
    _apply_sink(objects, sink_words, float_words)
    _apply_open_shut(objects, open_words, shut_words, float_words)
    _apply_hot_melt(objects, hot_words, melt_words, float_words)
    _apply_defeat(objects, you_words, defeat_words, float_words)
    _apply_win(next_state, objects, you_words, win_words, float_words)
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
        for (dx, dy) in directions:
            subjects = []
            cx = x
            cy = y
            expect_subject = True
            while True:
                current = text_at.get((cx, cy))
                if not isinstance(current, dict):
                    subjects = []
                    break
                current_type = current.get('type')
                current_word = current.get('word')
                if expect_subject:
                    if current_type != 'rule_noun' or not isinstance(current_word, str):
                        subjects = []
                        break
                    if current_word not in subjects:
                        subjects.append(current_word)
                    expect_subject = False
                    cx += dx
                    cy += dy
                    continue
                if current_type == 'rule_operator' and current_word == 'and':
                    expect_subject = True
                    cx += dx
                    cy += dy
                    continue
                if current_type == 'rule_operator' and current_word == 'is':
                    cx += dx
                    cy += dy
                    break
                subjects = []
                break
            if not subjects:
                continue
            expect_value = True
            found_value = False
            while True:
                current = text_at.get((cx, cy))
                if not isinstance(current, dict):
                    break
                current_type = current.get('type')
                current_word = current.get('word')
                if expect_value:
                    if not isinstance(current_word, str):
                        break
                    if current_type != 'rule_property' and current_type != 'rule_noun':
                        break
                    for subject in subjects:
                        if subject not in rules:
                            rules[subject] = {'properties': set(), 'nouns': []}
                        if current_type == 'rule_property':
                            rules[subject]['properties'].add(current_word)
                        elif current_word not in rules[subject]['nouns']:
                            rules[subject]['nouns'].append(current_word)
                    found_value = True
                    expect_value = False
                    cx += dx
                    cy += dy
                    continue
                if current_type == 'rule_operator' and current_word == 'and':
                    expect_value = True
                    cx += dx
                    cy += dy
                    continue
                break
            if found_value:
                continue
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

def _attempt_move(obj, dx, dy, grid_size, position_objects, push_words, stop_words, open_words, shut_words):
    if not isinstance(obj, dict):
        return False
    facing = _delta_to_facing(dx, dy)
    if facing is not None:
        obj['direction'] = facing
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
    obj_word = obj.get('word')
    obj_is_open = obj.get('type') == 'world_object' and isinstance(obj_word, str) and (obj_word in open_words)
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
            blocker_word = blocker.get('word')
            if not (obj_is_open and isinstance(blocker_word, str) and (blocker_word in shut_words)):
                return False
    for blocker in pushables:
        if not _attempt_move(blocker, dx, dy, grid_size, position_objects, push_words, stop_words, open_words, shut_words):
            return False
    _relocate_object(obj, target, position_objects)
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

def _apply_move_properties(objects, grid_size, position_objects, push_words, stop_words, move_words, open_words, shut_words):
    if not isinstance(objects, list):
        return
    mover_groups = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        if not isinstance(word, str) or word not in move_words:
            continue
        facing = obj.get('direction')
        if not isinstance(facing, str):
            facing = 'facing right'
            obj['direction'] = facing
        if facing not in mover_groups:
            mover_groups[facing] = []
        mover_groups[facing].append(obj)
    for facing in mover_groups:
        (dx, dy) = _facing_to_delta(facing)
        for obj in _ordered_movers(mover_groups.get(facing, []), dx, dy):
            current_facing = obj.get('direction')
            if not isinstance(current_facing, str):
                current_facing = facing
                obj['direction'] = current_facing
            (move_dx, move_dy) = _facing_to_delta(current_facing)
            moved = False
            if move_dx != 0 or move_dy != 0:
                moved = _attempt_move(obj, move_dx, move_dy, grid_size, position_objects, push_words, stop_words, open_words, shut_words)
            if not moved:
                obj['direction'] = _opposite_facing(current_facing)

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

def _apply_open_shut(objects, open_words, shut_words, float_words):
    if not isinstance(objects, list):
        return
    if not isinstance(open_words, set) or not isinstance(shut_words, set):
        return
    if not open_words or not shut_words:
        return
    position_objects = _collect_position_objects(objects)
    remove_ids = set()
    for same_cell in position_objects.values():
        if not isinstance(same_cell, list):
            continue
        for is_float in (False, True):
            has_open = False
            has_shut = False
            group = []
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') != 'world_object':
                    continue
                if _is_float_object(obj, float_words) != is_float:
                    continue
                word = obj.get('word')
                if not isinstance(word, str):
                    continue
                group.append(obj)
                if word in open_words:
                    has_open = True
                if word in shut_words:
                    has_shut = True
            if not (has_open and has_shut):
                continue
            for obj in group:
                word = obj.get('word')
                if isinstance(word, str) and (word in open_words or word in shut_words):
                    remove_ids.add(id(obj))
    if not remove_ids:
        return
    kept = []
    for obj in objects:
        if id(obj) not in remove_ids:
            kept.append(obj)
    objects[:] = kept

def _apply_sink(objects, sink_words, float_words):
    if not isinstance(objects, list):
        return
    if not isinstance(sink_words, set) or not sink_words:
        return
    position_objects = _collect_position_objects(objects)
    remove_ids = set()
    for same_cell in position_objects.values():
        if not isinstance(same_cell, list) or len(same_cell) < 2:
            continue
        for is_float in (False, True):
            sink_count = 0
            total_count = 0
            group = []
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                word = obj.get('word')
                if not isinstance(word, str):
                    continue
                if _is_float_entity(obj, float_words) != is_float:
                    continue
                group.append(obj)
                total_count += 1
                if obj.get('type') == 'world_object' and word in sink_words:
                    sink_count += 1
            if sink_count < 1 or total_count < 2:
                continue
            for obj in group:
                remove_ids.add(id(obj))
    if not remove_ids:
        return
    kept = []
    for obj in objects:
        if id(obj) not in remove_ids:
            kept.append(obj)
    objects[:] = kept

def _apply_defeat(objects, you_words, defeat_words, float_words):
    if not isinstance(objects, list):
        return
    if not isinstance(you_words, set) or not isinstance(defeat_words, set):
        return
    if not you_words or not defeat_words:
        return
    position_objects = _collect_position_objects(objects)
    remove_ids = set()
    for same_cell in position_objects.values():
        if not isinstance(same_cell, list):
            continue
        for is_float in (False, True):
            has_you = False
            has_defeat = False
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') != 'world_object':
                    continue
                if _is_float_object(obj, float_words) != is_float:
                    continue
                word = obj.get('word')
                if not isinstance(word, str):
                    continue
                if word in you_words:
                    has_you = True
                if word in defeat_words:
                    has_defeat = True
            if not (has_you and has_defeat):
                continue
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') != 'world_object':
                    continue
                if _is_float_object(obj, float_words) != is_float:
                    continue
                word = obj.get('word')
                if isinstance(word, str) and word in you_words:
                    remove_ids.add(id(obj))
    if not remove_ids:
        return
    kept = []
    for obj in objects:
        if id(obj) not in remove_ids:
            kept.append(obj)
    objects[:] = kept

def _is_float_object(obj, float_words):
    if not isinstance(obj, dict):
        return False
    if obj.get('type') != 'world_object':
        return False
    if not isinstance(float_words, set):
        return False
    word = obj.get('word')
    return isinstance(word, str) and word in float_words

def _apply_noun_transformations(objects, rules):
    changed = False
    if not isinstance(objects, list):
        return changed
    if not isinstance(rules, dict):
        return changed
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
        if not isinstance(target_word, str) or target_word == word:
            continue
        obj['word'] = target_word
        changed = True
    return changed

def _is_float_entity(obj, float_words):
    if not isinstance(obj, dict):
        return False
    if obj.get('type') == 'world_object':
        return _is_float_object(obj, float_words)
    if not isinstance(float_words, set):
        return False
    if 'text' not in float_words:
        return False
    word = obj.get('word')
    return isinstance(word, str)

def _extract_property_word_sets(rules):
    result = {}
    if not isinstance(rules, dict):
        return result
    for (subject, subject_rules) in rules.items():
        if not isinstance(subject_rules, dict):
            continue
        properties = subject_rules.get('properties')
        if not isinstance(properties, set):
            continue
        for prop in properties:
            if not isinstance(prop, str):
                continue
            if prop not in result:
                result[prop] = set()
            result[prop].add(subject)
    return result

def _apply_win(next_state, objects, you_words, win_words, float_words):
    if not isinstance(next_state, dict):
        return
    if not isinstance(objects, list):
        return
    if not isinstance(you_words, set) or not isinstance(win_words, set):
        return
    if not you_words or not win_words:
        return
    position_objects = _collect_position_objects(objects)
    won = False
    for same_cell in position_objects.values():
        if not isinstance(same_cell, list):
            continue
        for is_float in (False, True):
            has_you = False
            has_win = False
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') != 'world_object':
                    continue
                if _is_float_object(obj, float_words) != is_float:
                    continue
                word = obj.get('word')
                if not isinstance(word, str):
                    continue
                if word in you_words:
                    has_you = True
                if word in win_words:
                    has_win = True
            if has_you and has_win:
                won = True
                break
        if won:
            break
    if not won:
        return
    step = next_state.get('step')
    if not isinstance(step, dict):
        step = {}
        next_state['step'] = step
    step['terminated'] = True

def _ordered_movers(objects, dx, dy):
    movers = []
    if not isinstance(objects, list):
        return movers
    indexed = []
    index = 0
    for obj in objects:
        if isinstance(obj, dict):
            position = obj.get('position')
            if isinstance(position, list) and len(position) == 2:
                x = position[0]
                y = position[1]
                if isinstance(x, int) and isinstance(y, int):
                    indexed.append((index, x, y, obj))
        index += 1
    if dy > 0:
        indexed.sort(key=lambda item: (-item[2], item[0]))
    elif dy < 0:
        indexed.sort(key=lambda item: (item[2], item[0]))
    elif dx > 0:
        indexed.sort(key=lambda item: (-item[1], item[0]))
    elif dx < 0:
        indexed.sort(key=lambda item: (item[1], item[0]))
    else:
        indexed.sort(key=lambda item: item[0])
    for item in indexed:
        movers.append(item[3])
    return movers

def _apply_hot_melt(objects, hot_words, melt_words, float_words):
    if not isinstance(objects, list):
        return
    if not isinstance(hot_words, set) or not isinstance(melt_words, set):
        return
    if not hot_words or not melt_words:
        return
    position_objects = _collect_position_objects(objects)
    remove_ids = set()
    for same_cell in position_objects.values():
        if not isinstance(same_cell, list):
            continue
        for is_float in (False, True):
            has_hot = False
            melt_ids = []
            for obj in same_cell:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') != 'world_object':
                    continue
                if _is_float_object(obj, float_words) != is_float:
                    continue
                word = obj.get('word')
                if not isinstance(word, str):
                    continue
                if word in hot_words:
                    has_hot = True
                if word in melt_words:
                    melt_ids.append(id(obj))
            if not has_hot or not melt_ids:
                continue
            for obj_id in melt_ids:
                remove_ids.add(obj_id)
    if not remove_ids:
        return
    kept = []
    for obj in objects:
        if id(obj) not in remove_ids:
            kept.append(obj)
    objects[:] = kept
