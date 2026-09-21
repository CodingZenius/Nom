import unittest
from nomad.util import extract_json, trunc
from nomad.agent import parse_action


class TestJson(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_around(self):
        self.assertEqual(extract_json('Sure! Here you go: {"tool":"ls","args":{}} hope it helps'),
                         {"tool": "ls", "args": {}})

    def test_raw_newline_in_string(self):
        v = extract_json('{"tool":"write_file","args":{"path":"a.py","content":"print(1)\nprint(2)"}}')
        self.assertEqual(v["args"]["content"], "print(1)\nprint(2)")

    def test_trailing_comma_and_python_dict(self):
        self.assertEqual(extract_json('{"a": [1,2,],}'), {"a": [1, 2]})
        self.assertEqual(extract_json("{'a': True, 'b': None}"), {"a": True, "b": None})

    def test_braces_inside_strings(self):
        v = extract_json('junk {"content": "def f():\\n  return {1: 2}"} tail')
        self.assertIn("return {1: 2}", v["content"])

    def test_garbage(self):
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json(""))

    def test_parse_action_flattened(self):
        a = parse_action('{"thought":"x","tool":"done","summary":"all good"}')
        self.assertEqual(a["tool"], "done")
        self.assertEqual(a["args"]["summary"], "all good")
        self.assertIsNone(parse_action('{"thought":"no tool"}'))

    def test_trunc(self):
        self.assertTrue(len(trunc("x" * 5000, 100)) < 200)


if __name__ == "__main__":
    unittest.main()
