from src.example_module.example_code import ExampleClass
import unittest


class TestExampleClass(unittest.TestCase):

    def test_sum(self):
        example = ExampleClass()
        result = example.sum(2, 2)

        self.assertEqual(result, 4)

