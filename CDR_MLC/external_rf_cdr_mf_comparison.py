"""Generic external-dataset RF, Original CDR-MLC, and MF-CDR-MLC runner.

This stable entry point delegates to the leakage-safe implementation shared by
SDNCampus, ISCX-Tor, ISCX-VPN, and UNSW-IoT. Dataset-specific feature aliases,
fixed stratified splits, and train-only preprocessing are recorded in each
run manifest.
"""
from __future__ import annotations

from sdncampus_rf_cdr_mf_comparison import main


if __name__ == "__main__":
    main()
