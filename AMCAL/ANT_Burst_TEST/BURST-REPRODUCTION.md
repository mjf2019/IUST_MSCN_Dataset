# AdvBurst reconstruction and integrity checks

Source: Sadeghzadeh, Shiravi and Jalili, *Adversarial Network Traffic: Towards Evaluating the Robustness of Deep Learning-Based Network Traffic Classification*, arXiv:2003.01261v4, Algorithm 3 and Section V.D.

AdvBurst optimizes a class-conditional universal vector by raw-gradient ascent on frozen CNN cross entropy, not sign-gradient FGSM. It appends dummy packet IATs to a selected same-direction burst and truncates/pads the classifier sequence to 100 packets. The FTSC-IAT experiment chooses the first destination-to-source burst, uses 2000 iterations, batch size 64, step 0.01 and dummy IAT bounds 0.001–0.1 seconds. Equation 21 uses microseconds: `2*log(iat_us+1)/log(iat_max_us)`; preserve the packet direction sign. These attack settings are distinct from AMCAL's Context optimizer learning rate.

## Existing CSV audit

```powershell
python AMCAL/amcal_burst_reproduce.py audit --data-dir AMCAL/ANT_Burst_TEST/Dataset/AdvBurst_FTSC-IAT --levels 1 3 5 7 10 12 15 17 20 --output AMCAL/ANT_Burst_TEST/protocol_reports/burst_integrity_local
```

Use a new output directory. The audit checks matching schema/labels and whether each attacked row can be exactly reconstructed by insertion and suffix truncation from the corresponding level-0 row. It records ambiguous/unchanged rows and whether uniquely recovered dummy vectors are shared per class. This is stronger evidence for positional correspondence than matching labels alone, but does not prove capture/session provenance, selected burst correctness or gradient optimization. Unchanged rows can result from insertion beyond the visible 100-packet window, missing eligible bursts, or other causes; they require provenance review, not automatic rejection.

## Generate an independent sequence-space attack

Requires two disjoint CSVs in the explicitly confirmed signed-log IAT representation, a matching common-protocol CNN checkpoint, and packet metadata for each CSV. Metadata columns:

- `source_row`: globally unique original flow ID, disjoint between fit and target.
- `packet_count`: actual visible packet count (1–100), before zero padding.
- `burst_end`: exclusive packet index at the end of the selected first destination-to-source burst.
- `direction`: +1 or -1 according to the original source/destination convention.

Do not infer these metadata from standardized signs, labels or row numbers. Supply them from flow extraction or the original attack generator. Class-conditional UAP generation uses known class labels, matching the source attack setting; this is an attacker-side assumption, not an AMCAL deployment capability.

```powershell
python AMCAL/amcal_burst_reproduce.py generate --fit-csv FIT_SIGNED_LOG.csv --target-csv TARGET_SIGNED_LOG.csv --fit-metadata FIT_PACKET_METADATA.csv --target-metadata TARGET_PACKET_METADATA.csv --checkpoint-output MATCHING_COMMON_PROTOCOL_RUN --representation signed-log --iat-max-seconds 180 --levels 1 3 5 7 10 12 15 17 20 --iterations 2000 --batch-size 64 --step-size 0.01 --seed 42 --output NEW_ATTACK_OUTPUT
```

The 180-second maximum reproduces the paper setting; recover the actual saved maximum for the user's dataset before claiming numerical reproduction. Generation validates the classifier preprocessing hash, input schema and fit/target separation; existing data is never overwritten. Outputs include attack and one-run random-control CSVs, UAP vectors, hashes and configuration. This reproduces the supplied classifier's sequence-space attack, not the authors' 150-model architecture search or packet-level replay. The random control uses one repetition, whereas the paper reports 50.

Local checks: syntax and insertion logic were tested; existing CSV structural audits ran on CPU. GPU gradient generation has not been executed in this environment.
