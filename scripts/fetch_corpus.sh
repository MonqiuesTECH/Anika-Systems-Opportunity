#!/usr/bin/env bash
set -euo pipefail

download() {
  url="$1"; out="$2"
  echo "-> $out"
  curl -sSL "$url" -o "$out"
}

mkdir -p data/raw/{anika,policy,gao,gsa,vehicles}

# ---------- Anika Systems (core docs)
download "https://www.anikasystems.com/"                                "data/raw/anika/anika-home.html"
download "https://www.anikasystems.com/whoweare.html"                    "data/raw/anika/who-we-are.html"
download "https://www.anikasystems.com/capabilities.html"                "data/raw/anika/capabilities.html"
download "https://www.anikasystems.com/contracts.html"                   "data/raw/anika/contracts.html"
download "https://www.anikasystems.com/insights.html"                    "data/raw/anika/insights.html"

# ---------- Federal AI / policy docs
download "https://www.whitehouse.gov/wp-content/uploads/2024/03/M-24-10-Advancing-Governance-Innovation-and-Risk-Management-for-Agency-Use-of-Artificial-Intelligence.pdf" "data/raw/policy/OMB-M-24-10.pdf"
download "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"        "data/raw/policy/NIST-AI-RMF-1.0.pdf"
download "https://www.govinfo.gov/content/pkg/DCPD-202300949/pdf/DCPD-202300949.pdf" "data/raw/policy/EO-14110.pdf"
download "https://media.defense.gov/2024/Apr/15/2003439257/-1/-1/0/CSI-DEPLOYING-AI-SYSTEMS-SECURELY.PDF" "data/raw/policy/CISA-Deploying-AI-Securely.pdf"
download "https://www.dhs.gov/sites/default/files/2024-04/24_0426_dhs_ai-ci-safety-security-guidelines-508c.pdf" "data/raw/policy/DHS-AI-Guidelines.pdf"

# ---------- GAO reports
download "https://www.gao.gov/assets/gao-24-107332.pdf"                  "data/raw/gao/GAO-24-107332.pdf"
download "https://www.gao.gov/assets/gao-25-107653.pdf"                  "data/raw/gao/GAO-25-107653.pdf"

# ---------- GSA AI resources
download "https://coe.gsa.gov/coe/ai-guide-for-government/print-all/index.html" "data/raw/gsa/GSA-AI-Guide.html"
download "https://coe.gsa.gov/docs/2020/AIServiceCatalog.pdf"            "data/raw/gsa/GSA-AI-Service-Catalog.pdf"

# ---------- Contract vehicles
download "https://www.nitaac.nih.gov/gwacs/cio-sp3"                      "data/raw/vehicles/CIO-SP3.html"
download "https://itvmo.gsa.gov/it-vehicles/"                            "data/raw/vehicles/GSA-IT-Vehicles.html"

echo "✅ Corpus fetched. Files saved to data/raw/"
