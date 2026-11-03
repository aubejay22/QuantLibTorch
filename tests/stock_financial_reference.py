"""Independent stock QuantLib market bumps on fixed original Heston meshes."""
import copy
import json
import sys
from stock_latent_reference import price


def evaluate(request):
    h = 1e-5
    result = {"price": price(request), "gradients": []}
    keys = ([("discount_curve", i) for i in (1, 2)] +
            [("dividend_curve", i) for i in (1, 2)]) if request.get("discount_curve") else [
                ("rate", None), ("dividend_yield", None)]
    for key, index in keys:
        plus, minus = copy.deepcopy(request), copy.deepcopy(request)
        if index is None:
            plus[key] += h
            minus[key] -= h
        else:
            plus[key][index][1] += h
            minus[key][index][1] -= h
        result["gradients"].append((price(plus)-price(minus))/(2*h))
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(json.load(sys.stdin))))
