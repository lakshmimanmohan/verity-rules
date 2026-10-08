"""Enables `python -m verity_rules`."""
import sys

from .cli import main

sys.exit(main())
