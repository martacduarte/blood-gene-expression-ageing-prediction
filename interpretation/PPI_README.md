# PPI network analysis

The protein-protein interaction (PPI) network analysis was performed using the STRING web interface
(https://string-db.org), not a standalone script.

## Input

The input to STRING was the 200-gene signature selected by Ridge
regression from the DGE-filtered gene space:
genes were first restricted to those identified as age-associated by the
limma-voom differential expression analysis (`differential_expression/`), and the 200 genes carried forward to the regression models
were then selected from that DGE-filtered set by Ridge-based ranking (`feature_selection/`).

## How it was done

1. The 200 Ridge-selected genes were submitted to STRING as a
   multiple-protein search.
2. **Note on gene ID format:** STRING does not always accept versioned
   Ensembl gene IDs (e.g. `ENSG00000182263.14`). The version suffix
   (`.14`) had to be removed, leaving the unversioned ID
   (`ENSG00000182263`), for STRING to correctly resolve all genes.
3. Default STRING parameters were used to identify known and predicted
   protein-protein interactions among the signature genes.

No code was used for this step.
