import unittest
from learn.hello import get_hello

class HelloTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual('Hello world', get_hello())

if __name__ == "__main__":
    unittest.main()
