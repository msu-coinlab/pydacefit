"""Smoke test: execute pydacefit/usage.py end-to-end with a headless backend."""

import os
import unittest


class UsageTest(unittest.TestCase):
    def test(self):

        USAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "pydacefit")

        print(USAGE_DIR)
        for fname in ["usage.py"]:
            with open(os.path.join(USAGE_DIR, fname)) as f:
                s = f.read()

                no_plots = "import matplotlib\nimport matplotlib.pyplot\nmatplotlib.use('Agg')\n"

                s = no_plots + s

                s += "\nmatplotlib.pyplot.close()\n"

                try:
                    exec(s, globals())
                except Exception as e:
                    raise Exception("Usage %s failed." % fname) from e


if __name__ == "__main__":
    unittest.main()
