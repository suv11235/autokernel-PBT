candidates: 120
distinct sources: {'softmax': 36, 'layernorm': 49}
the hidden checker, re-run, calls wrong: []
the other bundle rejects: []
worst relative difference from stock: {'softmax': (5.820766091346741e-07, 'completed-r1/05'), 'layernorm': (2.0685545678134076e-05, 'completed-r1/16')}
branches: [('x.size == 0', 2), ('x.ndim < 2 or n == 0 or x.size <= (1 << 17)', 1), ('x.ndim', 1), ('ones is None', 1), ('x.ndim < 2 or x.size * x.itemsize <= (1 << 19) or n == 0', 1), ('x.ndim < 2 or rows <= block or x.dtype != np.float32', 1), ('n', 1), ('x.ndim < 2 or n == 0 or x.size <= (1 << 18)', 1)]
one source measured 3+ times: 26 groups
max/min: median 1.058, widest (1.22855956799644, 'current-r2 layernorm: 1.95 to 2.40')
groups past the 2% margin: 22; past 10%: 8
