"""Diagnostic for exp11 (declared post hoc, not prespecified).

Does the Under-operator's PET SHARK advantage depend on hidden assumption H4,
the functional form of PET's role face?  Re-run part 2 with the size x aquatic
interaction term removed from the role features.
"""
import json
import numpy as np
import exp11_operators as E

out = {}
for name, rf in [("with size*aquatic term", E.role_features),
                 ("linear attributes only", lambda v: np.array([v[0], v[1], v[2], v[3], 1.0]))]:
    E.role_features = rf
    vals = [E.part2(s)["shark"]["Under-operator"] for s in range(E.N_SEEDS)]
    out[name] = np.mean(vals, 0).tolist()
    print(f"  {name:24s} Under(PET, SHARK) -> housing {out[name][0]:.2f}, risk {out[name][1]:.2f}   (true 2.70, 0.95)")
json.dump(out, open("results_exp11b.json", "w"))
