# scripts/fetch_corpus.py
import os
from pathlib import Path
from typing import List, Tuple
import requests

# [(url, relative_output_path)]
URLS: List[Tuple[str, str]] = [
    # ---- Anika Systems (core pages)
    ("https://www.anikasystems.com/",                           "anika/anika-home.html"),
    ("https://www.anikasystems.com/whoweare.html",              "anika/who-we-are.html"),
    ("https://www.anikasystems.com/capabilities.html",          "anika/capabilities.html"),
    ("https://www.anikasystems.com/contracts.html",             "anika/contracts.html"),
    ("https://www.anikasystems.com/insights.html",              "anika/insights.html"),
    # ---- Federal AI / policy docs
    ("https://www.whitehouse.gov/wp-content/uploads/2024/03/M-24-10-Advancing-Governance-Innovation-and-Risk-Management-for-Agency-Use-of-Artificial-Intelligence.pdf", "policy/OMB-M-24-10.pdf"),
    ("https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",  "policy/NIST-AI-RMF-1.0.pdf"),
    ("https://www.govinfo.gov/content/pkg/DCPD-202300949/pdf/DCPD-202300949.pdf", "policy/EO-14110.pdf"),
    ("https://media.defense.gov/2024/Apr/15/2003439257/-1/-1/0/CSI-DEPLOYING-AI-SYSTEMS-SECURELY.PDF", "policy/CISA-Deploying-AI-Securely.pdf"),
    ("https://www.dhs.gov/sites/default/files/2024-04/24_0426_dhs_ai-ci-safety-security-guidelines-508c.pdf", "policy/DHS-AI-Guidelines.pdf"),
    # ---- GAO reports
    ("https://www.gao.gov/assets/gao-24-107332.pdf",           "gao/GAO-24-107332.pdf"),
    ("https://www.gao.gov/assets/gao-25-107653.pdf",           "gao/GAO-25-107653.pdf"),
    # ---- GSA AI resources
    ("https://coe.gsa.gov/coe/ai-guide-for-government/print-all/index.html", "gsa/GSA-AI-Guide.html"),
    ("https://coe.gsa.gov/docs/2020/AIServiceCatalog.pdf",     "gsa/GSA-AI-Service-Catalog.pdf"),
    # ---- Contract vehicles
    ("https://www.nitaac.nih.gov/gwacs/cio-sp3",               "vehicles/CIO-SP3.html"),
    ("https://itvmo.gsa.gov/it-vehicles/",                     "vehicles/GSA-IT-Vehicles.html"),
]

def fetch_starter_corpus(raw_root: Path = Path("data/raw")) -> int:
    raw_root.mkdir(parents=True, exist_ok=True)
    saved = 0
    for url, rel_path in URLS:
        out_path = raw_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            # Binary write is safe for both HTML/PDF
            out_path.write_bytes(resp.content)
            saved += 1
        except Exception as e:
            # Keep going; just skip failed URL
            print(f"[fetch] skip {url}: {e}")
    return saved
#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
python - <<'PY'
from src.loaders import fetch_starter_corpus
from pathlib import Path
saved, failed = fetch_starter_corpus(Path("data/raw"), target_count=40)
print(f"Saved={len(saved)} Failed={len(failed)}")
PY
echo "Done."
