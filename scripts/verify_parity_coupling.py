"""Finite exact checks for higher-order parity coupling."""
import json
from pathlib import Path

import torch

from wm_security.parity_coupling import EVEN, ODD, parity_index_tables


def pair_table(table, i, j):
    return sorted((int(x), int(y)) for x, y in zip(table[:, i], table[:, j]))


def main():
    even, odd = parity_index_tables()
    assert pair_table(even, 0, 1) == pair_table(odd, 0, 1)
    assert pair_table(even, 0, 2) == pair_table(odd, 0, 2)
    assert pair_table(even, 1, 2) == pair_table(odd, 1, 2)
    even_product = torch.prod(1 - 2 * even, dim=1).tolist()
    odd_product = torch.prod(1 - 2 * odd, dim=1).tolist()
    assert sum(even_product) == 4 and sum(odd_product) == -4
    result = {
        'status': 'completed',
        'marginal_index_multiset': [[0, 0, 1, 1]] * 3,
        'pairwise_index_tables_equal': True,
        'even_triple_product_mean': sum(even_product) / 4,
        'odd_triple_product_mean': sum(odd_product) / 4,
        'interpretation': 'The construction changes only third-order parity; it is an exact finite-bank witness, not task evidence.'
    }
    out = Path('runs/parity_coupling_exact_v0'); out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
