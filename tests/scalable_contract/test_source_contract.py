import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parsed(relative):
    return ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def function_node(tree, class_name, function_name):
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in owner.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name)


def names(node):
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


class SourceContractTests(unittest.TestCase):
 def test_prefix_is_explicitly_four_stages(self):
    tree = parsed("scalable_attribute/canonical/prefix.py")
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == "FrozenUnicornPrefix")
    assignment = next(node for node in owner.body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name)
                              and target.id == "residual_stages"
                              for target in node.targets))
    self.assertIsInstance(assignment.value, ast.Constant)
    self.assertEqual(assignment.value.value, 4)


 def test_base_synthesis_inputs_are_prefix_state_only(self):
    tree = parsed("scalable_attribute/canonical/base_synthesis.py")
    forward = function_node(tree, "BaseSynthesis", "forward")
    self.assertEqual([argument.arg for argument in forward.args.args], ["self", "state"])
    attributes = {(item.value.id, item.attr) for item in ast.walk(forward)
                  if isinstance(item, ast.Attribute)
                  and isinstance(item.value, ast.Name)}
    self.assertLessEqual(attributes, {("self", "config"), ("self", "backbone"),
                                      ("state", "x4"), ("state", "f4"),
                                      ("state", "d4"), ("ME", "cat")})
    self.assertNotIn("attribute", names(forward))


 def test_base_reconstruction_has_no_gt_or_enhancement_dependency(self):
    tree = parsed("scalable_attribute/canonical/model.py")
    reconstruct = function_node(tree, "CanonicalBaseModel", "reconstruct_from_state")
    self.assertEqual([argument.arg for argument in reconstruct.args.args], ["self", "state"])
    used = names(reconstruct)
    self.assertNotIn("attribute", used)
    self.assertNotIn("ground_truth", used)
    self.assertNotIn("enhancement", used)


 def test_enhancement_decode_signature_has_no_ground_truth(self):
    tree = parsed("scalable_attribute/canonical/enhancement.py")
    decode = function_node(tree, "EnhancementVAE", "decode")
    arguments = [argument.arg for argument in decode.args.args]
    self.assertEqual(arguments, ["self", "payload", "base", "base_feature",
                                 "prior_dec", "embedding"])
    self.assertNotIn("ground_truth", names(decode))
    keywords = {item.arg for item in ast.walk(decode)
                if isinstance(item, ast.keyword)}
    self.assertNotIn("x_gt", keywords)


 def test_hard_path_uses_upstream_rate_convention(self):
    tree = parsed("scalable_attribute/canonical/scalable_model.py")
    hard = function_node(tree, "CanonicalScalableModel", "hard_reconstruct")
    calls = [node for node in ast.walk(hard) if isinstance(node, ast.Call)]
    len_calls = [node for node in calls if isinstance(node.func, ast.Name)
                 and node.func.id == "len"]
    self.assertEqual(len(len_calls), 1)
    constants = {node.value for node in ast.walk(hard)
                 if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    self.assertTrue({"strings", "min_v", "max_v", "base_bits", "full_bits",
                     "enhancement_bits", "prefix_rate"} <= constants)
    additions = [node for node in ast.walk(hard)
                 if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)]
    multiplications = [node for node in ast.walk(hard)
                       if isinstance(node, ast.BinOp)
                       and isinstance(node.op, ast.Mult)]
    self.assertTrue(additions)
    self.assertTrue(any(isinstance(node.right, ast.Constant)
                        and node.right.value == 8 for node in multiplications))
