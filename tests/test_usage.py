"""Smoke test: execute src/pydacefit/usage.py end-to-end with a headless backend."""

import os
import unittest


class UsageTest(unittest.TestCase):
    def test(self):

        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        USAGE_DIR = os.path.join(repo_root, "src", "pydacefit")

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
