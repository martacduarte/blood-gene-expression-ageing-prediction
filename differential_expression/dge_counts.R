suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
  library(optparse)
})

# Arguments
option_list <- list(
  make_option("--counts",   type="character"),
  make_option("--X_train",  type="character"),
  make_option("--X_test",   type="character"),
  make_option("--y_train",  type="character"),
  make_option("--y_test",   type="character"),
  make_option("--metadata", type="character"),
  make_option("--outdir",   type="character"),
  make_option("--fdr",      type="double",    default=0.05),
  make_option("--min_count",type="integer",   default=10L),
  make_option("--seed",     type="integer",   default=42L)
)
opt <- parse_args(OptionParser(option_list=option_list))
dir.create(opt$outdir, showWarnings=FALSE, recursive=TRUE)

set.seed(opt$seed)
cat(sprintf("Random seed set to: %d\n", opt$seed))

# Load TPM matrices
cat("\nLoading TPM matrices...\n")
X_train_tpm <- read.csv(opt$X_train, row.names=1, check.names=FALSE)
cat(sprintf("  X_train TPM: %d samples x %d genes\n",
            nrow(X_train_tpm), ncol(X_train_tpm)))

X_test_tpm <- read.csv(opt$X_test, row.names=1, check.names=FALSE)
cat(sprintf("  X_test TPM:  %d samples x %d genes\n",
            nrow(X_test_tpm), ncol(X_test_tpm)))

# Get train sample IDs
train_samples <- rownames(X_train_tpm)
test_samples  <- rownames(X_test_tpm)
all_samples   <- union(train_samples, test_samples)
cat(sprintf("  Total unique samples needed: %d\n", length(all_samples)))

# Preprocess raw counts
cat("\nLoading raw counts\n")

counts_raw <- read.table(
  gzfile(opt$counts),
  sep        = "\t",
  header     = TRUE,
  skip       = 2, 
  check.names = FALSE,
  row.names  = 1           # gene ID column 
)

cat(sprintf("  Raw counts loaded: %d genes x %d samples\n",
            nrow(counts_raw), ncol(counts_raw) - 1))

# Remove description column (second column)
counts_raw <- counts_raw[, -1, drop=FALSE]
cat(sprintf("  After removing Description col: %d genes x %d samples\n",
            nrow(counts_raw), ncol(counts_raw)))

# Filter counts to train samples only
cat("\nFiltering counts to training samples...\n")

# Match sample IDs
matched_train <- intersect(train_samples, colnames(counts_raw))
missing_train <- setdiff(train_samples, colnames(counts_raw))

if (length(missing_train) > 0) {
  cat(sprintf("  WARNING: %d train samples not found in counts file:\n",
              length(missing_train)))
  cat(paste("   ", missing_train[1:min(5, length(missing_train))], collapse="\n"), "\n")
}

cat(sprintf("  Matched %d / %d train samples in counts\n",
            length(matched_train), length(train_samples)))

counts_train <- counts_raw[, matched_train, drop=FALSE]
counts_train <- as.matrix(counts_train)
storage.mode(counts_train) <- "integer"

cat(sprintf("  Counts matrix for DGE: %d genes x %d samples\n",
            nrow(counts_train), ncol(counts_train)))

# Load metadata and y_train
cat("\nLoading metadata and labels...\n")
y_train <- read.csv(opt$y_train, stringsAsFactors=FALSE)
meta    <- read.csv(opt$metadata, stringsAsFactors=FALSE)

y_train$sample_id <- as.character(y_train$sample_id)
meta$sample_id    <- as.character(meta$sample_id)

# Align to matched_train samples
common <- sort(Reduce(intersect, list(
  matched_train,
  y_train$sample_id,
  meta$sample_id
)))
cat(sprintf("  Common samples after all alignment: %d\n", length(common)))

counts_train <- counts_train[, common, drop=FALSE]
y_train      <- y_train[match(common, y_train$sample_id), ]
meta         <- meta[match(common, meta$sample_id), ]

stopifnot(all(colnames(counts_train) == y_train$sample_id))
stopifnot(all(colnames(counts_train) == meta$sample_id))
cat("  Alignment verified\n")

# Low-count filtering
cat("\n Filtering low-expression genes...\n")
# Keep genes with at least min_count counts in at least 20% of samples
min_samples <- max(2, round(0.2 * ncol(counts_train)))
keep        <- rowSums(counts_train >= opt$min_count) >= min_samples
counts_filt <- counts_train[keep, , drop=FALSE]
cat(sprintf("  Kept %d / %d genes (min %d counts in >= %d samples)\n",
            sum(keep), nrow(counts_train), opt$min_count, min_samples))

# TMM normalisation + voom
cat("\nTMM normalisation and voom transformation...\n")

# Build DGEList and apply TMM normalisation
dge <- DGEList(counts=counts_filt)
dge <- calcNormFactors(dge, method="TMM")
cat(sprintf("  TMM normalisation applied\n"))
cat(sprintf("  Norm factors range: [%.3f, %.3f]\n",
            min(dge$samples$norm.factors),
            max(dge$samples$norm.factors)))

# Design matrix
sex <- ifelse(meta$SEX %in% c("Female", "FEMALE", "2"), 1, 0)
rin <- as.numeric(meta$SMRIN)
rin[is.na(rin)] <- median(rin, na.rm=TRUE)

design <- model.matrix(~ age + sex + rin,
                       data=data.frame(
                         age = y_train$age,
                         sex = sex,
                         rin = rin
                       ))
cat(sprintf("  Design matrix: %d samples x %d covariates\n",
            nrow(design), ncol(design)))
cat(sprintf("  Covariates: %s\n", paste(colnames(design), collapse=", ")))

# voom transformation
png(file.path(opt$outdir, "voom_plot.png"), width=1600, height=1200, res=200)
v <- voom(dge, design, plot=TRUE)
dev.off()
cat("  voom transformation applied \n")

# limma differential expression
cat("\n Running limma...\n")
fit     <- lmFit(v, design)
fit     <- eBayes(fit)
results <- topTable(fit, coef="age", number=Inf,
                    adjust.method="BH", sort.by="none")

# Filter significant genes
cat("\n Filtering significant genes...\n")

# Strip version suffixes from results rownames for matching
results$gene_base <- sub("\\..*$", "", rownames(results))

sig_mask  <- results$adj.P.Val < opt$fdr
sig_genes_versioned <- rownames(results)[sig_mask]  
sig_genes_base      <- results$gene_base[sig_mask]

cat(sprintf("  Significant (FDR < %.2f): %d / %d (%.1f%%)\n",
            opt$fdr,
            length(sig_genes_versioned),
            nrow(results),
            100 * length(sig_genes_versioned) / nrow(results)))

if (length(sig_genes_versioned) == 0) {
  cat("  WARNING: No significant genes. Using top 1000 by p-value.\n")
  top_idx             <- order(results$P.Value)[1:1000]
  sig_genes_versioned <- rownames(results)[top_idx]
  sig_genes_base      <- results$gene_base[top_idx]
}

# Filter TPM matrices using significant genes
cat("\n Filtering TPM matrices to significant genes...\n")

tpm_base_to_versioned <- setNames(
  colnames(X_train_tpm),
  sub("\\..*$", "", colnames(X_train_tpm))
)

matched_in_tpm  <- intersect(sig_genes_base, names(tpm_base_to_versioned))
missing_in_tpm  <- setdiff(sig_genes_base, names(tpm_base_to_versioned))

cat(sprintf("  Matched %d / %d sig genes in TPM matrix\n",
            length(matched_in_tpm), length(sig_genes_base)))
if (length(missing_in_tpm) > 0) {
  cat(sprintf("  WARNING: %d sig genes missing from TPM matrix\n",
              length(missing_in_tpm)))
}

tpm_cols_to_keep <- tpm_base_to_versioned[matched_in_tpm]


X_train_sig <- X_train_tpm[, tpm_cols_to_keep, drop=FALSE]
X_test_sig  <- X_test_tpm[,
                          intersect(tpm_cols_to_keep, colnames(X_test_tpm)),
                          drop=FALSE]

cat(sprintf("  X_train_dge: %d samples x %d genes\n",
            nrow(X_train_sig), ncol(X_train_sig)))
cat(sprintf("  X_test_dge:  %d samples x %d genes\n",
            nrow(X_test_sig), ncol(X_test_sig)))

# Save outputs
cat("\n Saving outputs...\n")

write.csv(X_train_sig, file.path(opt$outdir, "X_train_dge.csv"))
write.csv(X_test_sig,  file.path(opt$outdir, "X_test_dge.csv"))
file.copy(opt$y_train, file.path(opt$outdir, "y_train.csv"), overwrite=TRUE)
file.copy(opt$y_test,  file.path(opt$outdir, "y_test.csv"),  overwrite=TRUE)

# DGE results table
results_out <- data.frame(
  gene      = rownames(results),
  gene_base = results$gene_base,
  age_coef  = results$logFC,
  age_tstat = results$t,
  age_pval  = results$P.Value,
  age_fdr   = results$adj.P.Val
)
write.csv(results_out,
          file.path(opt$outdir, "dge_results.csv"),
          row.names=FALSE)

# Significant gene list 
writeLines(matched_in_tpm,
           file.path(opt$outdir, "sig_genes.txt"))

# Volcano plot
cat("Generating volcano plot...\n")
volcano_data             <- results_out
volcano_data$significant <- volcano_data$age_fdr < opt$fdr
volcano_data$neg_log10   <- -log10(volcano_data$age_fdr + 1e-300)

png(file.path(opt$outdir, "volcano_plot.png"),
    width=2400, height=1800, res=300)
plot(
  volcano_data$age_coef,
  volcano_data$neg_log10,
  col  = ifelse(volcano_data$significant, "#E74C3C", "#95A5A6"),
  pch  = 20, cex = 0.4,
  xlab = "Age Coefficient (log FC)",
  ylab = expression(-log[10](FDR)),
  main = sprintf("Volcano Plot: Age-Associated Genes (FDR < %.2f)", opt$fdr),
  cex.main = 1.2, cex.lab = 1.1
)
abline(h = -log10(opt$fdr), col="black", lty=2, lwd=1.5)
abline(v = 0,               col="grey40", lty=3, lwd=1)
n_sig   <- sum(volcano_data$significant, na.rm=TRUE)
n_total <- nrow(volcano_data)
legend("topright",
       legend = c(sprintf("Significant (n=%d)",     n_sig),
                  sprintf("Not significant (n=%d)", n_total - n_sig)),
       col    = c("#E74C3C", "#95A5A6"),
       pch=20, pt.cex=1.2, cex=0.9, bty="n")
dev.off()

cat(sprintf("\nDone"))