"""Train OGPO offline BC only and export a reusable flow checkpoint.

Use scripts/offline/train.sh TASK to inherit the exact OGPO task settings.
"""
from ogpo.main import main

if __name__ == '__main__':
    main(offline_only=True)
