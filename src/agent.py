"""Template entry point: `python src/agent.py "Your research question here"`.

Runs the full pipeline from the command line without installing the package (pip install -r requirements.txt is
enough). It is the same as the `research` CLI; all its options work here too, e.g.
  python src/agent.py "What has changed for first-line EGFR-mutant metastatic NSCLC since 2024-06-01?" --yes
  python src/agent.py --show 3
Without --yes it shows the brief and planned searches and asks before spending search credits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
