"""Score function calls with the official BFCL checker (package bfcl-eval, installed with --no-deps).

The checker module imports the leaderboard's table of model handlers only to decide whether dots in function names
were replaced by underscores (an OpenAI restriction); that table is replaced here by a stand-in that says no, so
none of the handlers' dependencies are needed. Outputs are parsed from the two formats of bfcl_constraints.py.
"""
import ast, json, os, sys, types
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_stub = types.ModuleType("bfcl_eval.constants.model_config")
_stub.MODEL_CONFIG_MAPPING = defaultdict(lambda: types.SimpleNamespace(underscore_to_dot=False))
sys.modules.setdefault("bfcl_eval.constants.model_config", _stub)
from bfcl_eval.constants.enums import Language                                   # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker             # noqa: E402
from bfcl_constraints import load, SPLITS, gold_text                             # noqa: E402


def _name(node):
    return _name(node.value) + "." + node.attr if isinstance(node, ast.Attribute) else node.id


def parse_calls(text, fmt):
    """Text in either output format -> [{function name: {argument: value}}, ...]; raises ValueError if it is not one."""
    try:
        if fmt == "json":
            calls = json.loads(text)
            return [{c["name"]: dict(c["arguments"])} for c in calls]
        tree = ast.parse(text.strip(), mode="eval").body
        return [{_name(c.func): {k.arg: ast.literal_eval(k.value) for k in c.keywords}} for c in tree.elts]
    except Exception as e:
        raise ValueError(f"{type(e).__name__}: {e}")


def score(text, fmt, question, answer, split):
    """True when the official checker accepts the call(s) in `text` for this example; False also when it does not parse."""
    try:
        out = parse_calls(text, fmt)
    except ValueError:
        return False
    try:
        return bool(ast_checker(question["function"], out, answer["ground_truth"], Language.PYTHON, split, "none")["valid"])
    except Exception:
        return False


if __name__ == "__main__":
    data = sys.argv[1]
    for c in SPLITS:
        q = load(f"{data}/BFCL_v3_{c}.json"); g = load(f"{data}/possible_answer/BFCL_v3_{c}.json")
        for fmt in ("python", "json"):
            ok = [score(gold_text(x["function"], y["ground_truth"], fmt), fmt, x, y, c) for x, y in zip(q, g)]
            print(f"{c:24s} {fmt:6s} reference text passes the official checker: {sum(ok):5d} of {len(ok):5d}", [x["id"] for x, o in zip(q, ok) if not o][:4])
