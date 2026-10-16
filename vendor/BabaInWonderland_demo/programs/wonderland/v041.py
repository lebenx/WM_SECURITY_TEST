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
    if not (isinstance(grid_size, list) and len(grid_size) == 2 and isinstance(grid_size[0], int) and isinstance(grid_size[1], int)):
        return next_state
    width = grid_size[0]
    height = grid_size[1]
    if width <= 0 or height <= 0:
        return next_state
    rules = _extract_rules(objects)
    properties = rules.get('properties') if isinstance(rules, dict) else {}
    transforms = rules.get('transforms') if isinstance(rules, dict) else {}
    chained_eat_words = _extract_chained_eat_words(objects)
    if not isinstance(properties, dict):
        properties = {}
    if not isinstance(transforms, dict):
        transforms = {}
    if not isinstance(chained_eat_words, set):
        chained_eat_words = set()
    _apply_transforms(objects, transforms)
    controllable_words = set()
    autonomous_words = set()
    pushable_words = set()
    blocking_words = {'hedge', 'door'}
    text_blocking_words = set()
    for (word, props) in properties.items():
        if not isinstance(props, set):
            continue
        if 'strange' in props:
            controllable_words.add(word)
        if 'drink' in props:
            autonomous_words.add(word)
        if 'grow' in props:
            pushable_words.add(word)
        if 'eat' in props:
            blocking_words.add(word)
            text_blocking_words.add(word)
    world_cells = {}
    text_cells = {}
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
        if obj.get('type') == 'world_object':
            items = world_cells.get(key)
            if items is None:
                world_cells[key] = [obj]
            else:
                items.append(obj)
        else:
            items = text_cells.get(key)
            if items is None:
                text_cells[key] = [obj]
            else:
                items.append(obj)
    background_words = {'tile', 'water'}
    world_map = {}
    text_blockers = set()
    blocked_text_cells = set()
    movable_words = set(controllable_words)
    movable_words.update(autonomous_words)
    movable_words.update(pushable_words)
    for (key, cell_objects) in world_cells.items():
        if not isinstance(cell_objects, list):
            continue
        candidates = []
        for obj in cell_objects:
            if not isinstance(obj, dict):
                continue
            word = obj.get('word')
            if word in background_words and word not in chained_eat_words:
                continue
            candidates.append(obj)
            if word in text_blocking_words:
                text_blockers.add(key)
        if not candidates:
            continue
        blocker = None
        for obj in candidates:
            word = obj.get('word')
            if word in pushable_words or word in controllable_words or word in autonomous_words or (word in blocking_words):
                blocker = obj
                break
        if blocker is not None:
            world_map[key] = blocker
            if blocker.get('word') not in movable_words:
                blocked_text_cells.add(key)
    blocked_text_cells.update(text_blockers)
    if action == 'idle':
        _move_world_objects(objects, autonomous_words, blocked_text_cells, text_cells, width, height, use_object_direction=True, world_map=world_map, pushable_words=pushable_words, special_collision_words=chained_eat_words, controllable_words=controllable_words)
    else:
        (dx, dy, facing) = _action_delta(action)
        if facing is None:
            return next_state
        if controllable_words:
            _move_world_objects(objects, controllable_words, blocked_text_cells, text_cells, width, height, dx=dx, dy=dy, facing=facing, use_object_direction=False, world_map=world_map, pushable_words=pushable_words, special_collision_words=chained_eat_words, controllable_words=controllable_words)
        if autonomous_words:
            _move_world_objects(objects, autonomous_words, blocked_text_cells, text_cells, width, height, use_object_direction=True, world_map=world_map, pushable_words=pushable_words, special_collision_words=chained_eat_words, controllable_words=controllable_words)
    _resolve_text_push_overlaps(objects, properties, controllable_words, autonomous_words, pushable_words, blocking_words, text_blocking_words, width, height)
    final_rules = _extract_rules(objects)
    final_properties = final_rules.get('properties') if isinstance(final_rules, dict) else {}
    final_transforms = final_rules.get('transforms') if isinstance(final_rules, dict) else {}
    if not isinstance(final_properties, dict):
        final_properties = {}
    if not isinstance(final_transforms, dict):
        final_transforms = {}
    _apply_transforms(objects, final_transforms)
    _update_step_terminated(next_state, objects, final_properties)
    _resolve_eat_interactions(objects, final_properties)
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
        items = position_map.get(key)
        if items is None:
            position_map[key] = [obj]
        else:
            items.append(obj)
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
            has_is = False
            for item in op_cell:
                if isinstance(item, dict) and item.get('type') == 'rule_operator' and (item.get('word') == 'is'):
                    has_is = True
                    break
            if not has_is:
                continue
            step = 2
            expect_target = True
            while True:
                cell = position_map.get((x + step * dx, y + step * dy), [])
                if expect_target:
                    found_target = False
                    for item in cell:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get('type')
                        item_word = item.get('word')
                        if not isinstance(item_word, str):
                            continue
                        if item_type == 'rule_property':
                            props = rules['properties'].get(word)
                            if props is None:
                                rules['properties'][word] = {item_word}
                            else:
                                props.add(item_word)
                            found_target = True
                        elif item_type == 'rule_noun':
                            targets = rules['transforms'].get(word)
                            if targets is None:
                                rules['transforms'][word] = {item_word}
                            else:
                                targets.add(item_word)
                            found_target = True
                    if not found_target:
                        break
                    expect_target = False
                else:
                    has_and = False
                    for item in cell:
                        if isinstance(item, dict) and item.get('type') == 'rule_operator' and (item.get('word') == 'and'):
                            has_and = True
                            break
                    if not has_and:
                        break
                    expect_target = True
                step += 1
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

def _push_text_chain(text_cells, world_occupied, x, y, dx, dy, width, height, moved_direction=None, world_map=None):
    if not isinstance(text_cells, dict):
        return True
    if not isinstance(world_map, dict):
        world_map = {}
    chain = []
    cx = x
    cy = y
    blocked = False
    while (cx, cy) in text_cells:
        chain.append((cx, cy))
        cx += dx
        cy += dy
        if cx < 0 or cy < 0 or cx >= width or (cy >= height):
            blocked = True
            break
        if (cx, cy) in world_occupied:
            occupant = world_map.get((cx, cy))
            if isinstance(occupant, dict):
                blocked = True
                break
    if blocked:
        if isinstance(moved_direction, str):
            for (px, py) in chain:
                items = text_cells.get((px, py))
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        item['direction'] = moved_direction
        return False
    index = len(chain) - 1
    while index >= 0:
        (px, py) = chain[index]
        items = text_cells.pop((px, py), None)
        if items:
            nx = px + dx
            ny = py + dy
            key = (nx, ny)
            existing = text_cells.get(key)
            if existing is None:
                text_cells[key] = items
            else:
                existing.extend(items)
            for item in items:
                if isinstance(item, dict):
                    item['position'] = [nx, ny]
                    item['_moved'] = True
                    if isinstance(moved_direction, str):
                        item['direction'] = moved_direction
        index -= 1
    return True

def _direction_delta(direction):
    if direction == 'facing up':
        return (0, -1)
    if direction == 'facing down':
        return (0, 1)
    if direction == 'facing left':
        return (-1, 0)
    if direction == 'facing right':
        return (1, 0)
    return (0, 0)

def _move_world_objects(objects, words, world_occupied, text_cells, width, height, dx=None, dy=None, facing=None, use_object_direction=False, world_map=None, pushable_words=None, special_collision_words=None, controllable_words=None):
    if not isinstance(objects, list):
        return
    if not isinstance(words, set) or not words:
        return
    if not isinstance(world_map, dict):
        world_map = {}
    if not isinstance(pushable_words, set):
        pushable_words = set()
    if not isinstance(special_collision_words, set):
        special_collision_words = set()
    if not isinstance(controllable_words, set):
        controllable_words = set()
    if not isinstance(world_occupied, set):
        world_occupied = set()
    movers = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') != 'world_object':
            continue
        word = obj.get('word')
        if word not in words:
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2):
            continue
        x = position[0]
        y = position[1]
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        move_dx = dx
        move_dy = dy
        moved_direction = facing
        if use_object_direction:
            moved_direction = obj.get('direction')
            (move_dx, move_dy) = _direction_delta(moved_direction)
        if not isinstance(move_dx, int) or not isinstance(move_dy, int):
            continue
        if facing is not None:
            obj['direction'] = facing
            moved_direction = facing
        priority = x * move_dx + y * move_dy
        movers.append((priority, obj, x, y, move_dx, move_dy, moved_direction))
    movers.sort(key=lambda item: item[0], reverse=True)
    for item in movers:
        obj = item[1]
        x = item[2]
        y = item[3]
        move_dx = item[4]
        move_dy = item[5]
        moved_direction = item[6]
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2 and (position[0] == x) and (position[1] == y)):
            continue
        if move_dx == 0 and move_dy == 0:
            continue
        nx = x + move_dx
        ny = y + move_dy
        blocked = False
        special_collision = False
        if nx < 0 or ny < 0 or nx >= width or (ny >= height):
            blocked = True
        else:
            target_obj = world_map.get((nx, ny))
            if isinstance(target_obj, dict):
                target_word = target_obj.get('word')
                mover_word = obj.get('word')
                if mover_word in controllable_words and target_word in special_collision_words and (target_word not in pushable_words):
                    obj['_remove_on_eat'] = True
                    target_obj['_remove_on_eat'] = True
                    blocked = True
                    special_collision = True
                elif not (target_obj.get('_moved') and target_word not in pushable_words):
                    if not _push_world_chain(world_map, text_cells, world_occupied, pushable_words, nx, ny, move_dx, move_dy, width, height, moved_direction=moved_direction):
                        blocked = True
            elif not _push_text_chain(text_cells, world_occupied, nx, ny, move_dx, move_dy, width, height, moved_direction=moved_direction, world_map=world_map):
                blocked = True
        if blocked:
            if use_object_direction:
                if moved_direction == 'facing up':
                    obj['direction'] = 'facing down'
                elif moved_direction == 'facing down':
                    obj['direction'] = 'facing up'
                elif moved_direction == 'facing left':
                    obj['direction'] = 'facing right'
                elif moved_direction == 'facing right':
                    obj['direction'] = 'facing left'
            elif special_collision:
                pass
            continue
        world_map.pop((x, y), None)
        obj['position'] = [nx, ny]
        obj['_moved'] = True
        world_map[nx, ny] = obj

def _push_world_chain(world_map, text_cells, world_occupied, pushable_words, x, y, dx, dy, width, height, moved_direction=None):
    if not isinstance(world_map, dict):
        return False
    if not isinstance(pushable_words, set) or not pushable_words:
        return False
    if not isinstance(world_occupied, set):
        world_occupied = set()
    chain = []
    cx = x
    cy = y
    blocked = False
    while True:
        obj = world_map.get((cx, cy))
        if not isinstance(obj, dict):
            break
        if obj.get('type') != 'world_object':
            blocked = True
            break
        if obj.get('word') not in pushable_words:
            blocked = True
            break
        chain.append(obj)
        cx += dx
        cy += dy
        if cx < 0 or cy < 0 or cx >= width or (cy >= height):
            blocked = True
            break
        next_obj = world_map.get((cx, cy))
        if isinstance(next_obj, dict) and next_obj.get('word') not in pushable_words:
            blocked = True
            break
    if blocked:
        if isinstance(moved_direction, str):
            for obj in chain:
                if isinstance(obj, dict):
                    obj['direction'] = moved_direction
        return False
    if not _push_text_chain(text_cells, world_occupied, cx, cy, dx, dy, width, height, moved_direction=moved_direction, world_map=world_map):
        if isinstance(moved_direction, str):
            for obj in chain:
                if isinstance(obj, dict):
                    obj['direction'] = moved_direction
        return False
    index = len(chain) - 1
    while index >= 0:
        obj = chain[index]
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2 and isinstance(position[0], int) and isinstance(position[1], int)):
            index -= 1
            continue
        ox = position[0]
        oy = position[1]
        nx = ox + dx
        ny = oy + dy
        world_map.pop((ox, oy), None)
        obj['position'] = [nx, ny]
        obj['_moved'] = True
        if isinstance(moved_direction, str):
            obj['direction'] = moved_direction
        world_map[nx, ny] = obj
        index -= 1
    return True

def _resolve_eat_interactions(objects, properties):
    if not isinstance(objects, list):
        return
    if not isinstance(properties, dict):
        properties = {}
    cells = {}
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2 and isinstance(position[0], int) and isinstance(position[1], int)):
            continue
        key = (position[0], position[1])
        items = cells.get(key)
        if items is None:
            cells[key] = [obj]
        else:
            items.append(obj)
    text_props = properties.get('text')
    text_is_wrong = isinstance(text_props, set) and 'wrong' in text_props
    background_words = {'water', 'tile'}
    remove_ids = set()
    for obj in objects:
        if isinstance(obj, dict) and obj.get('_remove_on_eat'):
            remove_ids.add(id(obj))
    for cell_objects in cells.values():
        if not isinstance(cell_objects, list) or len(cell_objects) < 2:
            continue
        has_moved_world = False
        has_moved_nonwrong_world = False
        has_text = False
        has_eater = False
        has_wake = False
        has_begin = False
        has_background_grin = False
        world_objects = []
        moved_strange_world_objects = []
        for obj in cell_objects:
            if not isinstance(obj, dict):
                continue
            moved = bool(obj.get('_moved'))
            if obj.get('type') == 'world_object':
                world_objects.append(obj)
                if moved:
                    has_moved_world = True
                word = obj.get('word')
                props = properties.get(word)
                if isinstance(word, str) and isinstance(props, set):
                    if 'eat' in props:
                        has_eater = True
                    if 'wake' in props:
                        has_wake = True
                    if 'begin' in props:
                        has_begin = True
                    if 'grin' in props and word in background_words:
                        has_background_grin = True
                    if moved and 'wrong' not in props:
                        has_moved_nonwrong_world = True
                    if moved and 'strange' in props:
                        moved_strange_world_objects.append(obj)
                elif moved:
                    has_moved_nonwrong_world = True
            else:
                has_text = True
        if has_begin and has_text and (not text_is_wrong):
            for obj in cell_objects:
                if isinstance(obj, dict):
                    remove_ids.add(id(obj))
            continue
        if len(world_objects) < 2:
            continue
        if has_moved_world and has_eater:
            for obj in world_objects:
                remove_ids.add(id(obj))
            continue
        if has_begin and has_moved_nonwrong_world:
            for obj in world_objects:
                remove_ids.add(id(obj))
            continue
        if has_background_grin:
            for obj in moved_strange_world_objects:
                word = obj.get('word')
                props = properties.get(word)
                if isinstance(word, str) and isinstance(props, set) and ('wrong' not in props):
                    remove_ids.add(id(obj))
        if has_wake:
            for obj in world_objects:
                word = obj.get('word')
                props = properties.get(word)
                if isinstance(word, str) and isinstance(props, set) and ('strange' in props):
                    remove_ids.add(id(obj))
    kept = []
    for obj in objects:
        if isinstance(obj, dict):
            obj.pop('_moved', None)
            obj.pop('_remove_on_eat', None)
        if id(obj) not in remove_ids:
            kept.append(obj)
    objects[:] = kept

def _apply_transforms(objects, transforms):
    if not isinstance(objects, list):
        return
    if not isinstance(transforms, dict):
        return
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

def _update_step_terminated(next_state, objects, properties):
    if not isinstance(next_state, dict):
        return
    terminated = False
    cells = {}
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get('type') != 'world_object':
                continue
            position = obj.get('position')
            if not (isinstance(position, list) and len(position) == 2 and isinstance(position[0], int) and isinstance(position[1], int)):
                continue
            key = (position[0], position[1])
            items = cells.get(key)
            if items is None:
                cells[key] = [obj]
            else:
                items.append(obj)
        for cell_objects in cells.values():
            if not isinstance(cell_objects, list) or len(cell_objects) < 2:
                continue
            has_shrink = False
            movable_strange = False
            for obj in cell_objects:
                word = obj.get('word')
                props = properties.get(word) if isinstance(properties, dict) else None
                if not isinstance(word, str) or not isinstance(props, set):
                    continue
                if 'shrink' in props:
                    has_shrink = True
                if obj.get('_moved') and 'strange' in props and ('wrong' not in props):
                    movable_strange = True
                if has_shrink and movable_strange:
                    terminated = True
                    break
            if terminated:
                break
    step = next_state.get('step')
    if isinstance(step, dict):
        step['terminated'] = terminated
    elif terminated:
        next_state['step'] = {'terminated': True}

def _resolve_text_push_overlaps(objects, properties, controllable_words, autonomous_words, pushable_words, blocking_words, text_blocking_words, width, height):
    if not isinstance(objects, list):
        return
    if not isinstance(pushable_words, set) or not pushable_words:
        return
    if not isinstance(controllable_words, set):
        controllable_words = set()
    if not isinstance(autonomous_words, set):
        autonomous_words = set()
    if not isinstance(blocking_words, set):
        blocking_words = set()
    if not isinstance(text_blocking_words, set):
        text_blocking_words = set()
    world_cells = {}
    text_cells = {}
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
        if obj.get('type') == 'world_object':
            items = world_cells.get(key)
            if items is None:
                world_cells[key] = [obj]
            else:
                items.append(obj)
        else:
            items = text_cells.get(key)
            if items is None:
                text_cells[key] = [obj]
            else:
                items.append(obj)
    background_words = {'tile', 'water'}
    world_map = {}
    blocked_text_cells = set()
    movable_words = set(controllable_words)
    movable_words.update(autonomous_words)
    movable_words.update(pushable_words)
    for (key, cell_objects) in world_cells.items():
        if not isinstance(cell_objects, list):
            continue
        candidates = []
        for obj in cell_objects:
            if not isinstance(obj, dict):
                continue
            word = obj.get('word')
            if word in background_words:
                continue
            candidates.append(obj)
        if not candidates:
            continue
        blocker = None
        for obj in candidates:
            word = obj.get('word')
            if word in pushable_words or word in controllable_words or word in autonomous_words or (word in blocking_words):
                blocker = obj
                break
        if blocker is not None:
            world_map[key] = blocker
            blocker_word = blocker.get('word')
            if blocker_word in text_blocking_words or blocker_word not in movable_words:
                blocked_text_cells.add(key)
    handled = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get('type') == 'world_object':
            continue
        if not obj.get('_moved'):
            continue
        direction = obj.get('direction')
        (dx, dy) = _direction_delta(direction)
        if dx == 0 and dy == 0:
            continue
        position = obj.get('position')
        if not (isinstance(position, list) and len(position) == 2 and isinstance(position[0], int) and isinstance(position[1], int)):
            continue
        x = position[0]
        y = position[1]
        key = (x, y, dx, dy)
        if key in handled:
            continue
        handled.add(key)
        target_obj = world_map.get((x, y))
        if not isinstance(target_obj, dict):
            continue
        if target_obj.get('word') not in pushable_words:
            continue
        _push_world_chain(world_map, text_cells, blocked_text_cells, pushable_words, x, y, dx, dy, width, height, moved_direction=direction)

def _extract_chained_eat_words(objects):
    result = set()
    if not isinstance(objects, list):
        return result
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
        items = position_map.get(key)
        if items is None:
            position_map[key] = [obj]
        else:
            items.append(obj)
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
            has_is = False
            for item in op_cell:
                if isinstance(item, dict) and item.get('type') == 'rule_operator' and (item.get('word') == 'is'):
                    has_is = True
                    break
            if not has_is:
                continue
            step = 2
            expect_target = True
            chain_props = set()
            while True:
                cell = position_map.get((x + step * dx, y + step * dy), [])
                if expect_target:
                    found_target = False
                    for item in cell:
                        if not isinstance(item, dict):
                            continue
                        if item.get('type') != 'rule_property':
                            continue
                        item_word = item.get('word')
                        if not isinstance(item_word, str):
                            continue
                        chain_props.add(item_word)
                        found_target = True
                    if not found_target:
                        break
                    expect_target = False
                else:
                    has_and = False
                    for item in cell:
                        if isinstance(item, dict) and item.get('type') == 'rule_operator' and (item.get('word') == 'and'):
                            has_and = True
                            break
                    if not has_and:
                        break
                    expect_target = True
                step += 1
            if 'late' in chain_props and 'eat' in chain_props:
                result.add(word)
    return result
