# PPI network analysis

The protein-protein interaction (PPI) network analysis described in Section 5.3
of the thesis was performed using the STRING web interface
(https://string-db.org), not a standalone script.

## Input

The input to STRING was the final 200-gene signature selected by Ridge
regression from the DGE-filtered gene space (Section 3.5.3 of the thesis):
genes were first restricted to those identified as age-associated by the
limma-voom differential expression analysis (`differential_expression/`,
Section 3.4), and the 200 genes carried forward to the regression models
were then selected from that DGE-filtered set by Ridge-based ranking
(`feature_selection/`, Section 3.5).

## How it was done

1. The 200 Ridge-selected genes were submitted to STRING as a
   multiple-protein search.
2. **Note on gene ID format:** STRING does not always accept versioned
   Ensembl gene IDs (e.g. `ENSG00000182263.14`). The version suffix
   (`.14`) had to be removed, leaving the unversioned ID
   (`ENSG00000182263`), for STRING to correctly resolve all genes.
3. Default STRING parameters were used to identify known and predicted
   protein-protein interactions among the signature genes.
4. The resulting network was exported and is shown in Figure 5.4 of the
   thesis, with two clusters of interest discussed in the accompanying
   text (the AGT/renin-angiotensin cluster and the NRCAM/SCN9A/PODXL2
   cluster).

No code was used for this step; it is documented here so the analysis can
be reproduced from the gene signature already produced earlier in the
pipeline (`feature_selection/`).
