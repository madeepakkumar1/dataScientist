import unittest
import robots


class TestRobots(unittest.TestCase):
    def test_url(self):
        self.assertEqual(True, robots.can_fetch('http://www.google.com'))
