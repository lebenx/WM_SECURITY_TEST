"""An exact native-code diagnostic, NOT evidence of LLM attack success."""
import json
from pathlib import Path
from wm_security.native import TypedRegistry
from world_model.modules.typed_registry import RegistryEntry

r=TypedRegistry()
r.merge([RegistryEntry('target.constraint','Action A preserves the target constraint.'),
         RegistryEntry('unrelated.constraint','Action B preserves an unrelated constraint.')])
trace=[]
for step in range(4):
    text,keys=r.compile_with_keys()
    trace.append({'step':step,'visible':sorted(keys),'evidence':{k:e.evidence for k,e in r.entries.items()}})
    r.credit_mismatch(keys)
result={'status':'code-path demonstration only; mismatch inputs here are manually supplied',
        'observation':'The updater does not condition rule penalties on which field or action was mispredicted.',
        'trace':trace}
Path('runs/credit_path').mkdir(parents=True,exist_ok=True)
Path('runs/credit_path/diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
