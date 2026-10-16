#John George - 090126#
#Data Preparation for ML#

setwd("F:/IOB/Projects/COVID-19_Ph.D/Transcriptomics_longCOVID_ML/CovRec_LC/ML_CovRec_LC_090126/")

library(DESeq2)
library(ggplot2)
library(limma)

# Load the count matrix
counts <- read.csv(file = "htseq_with_hgnc_all-6_CovRec-LC_080526.tsv", sep = "\t", header = TRUE, row.names = 1,  check.names = FALSE )
dim(counts)
str(counts[, 1:5])
ncol(counts)
condition <- factor(c(rep("Recovered", 96), rep("LC", 122)))
batch <- factor(c(rep(1, 12), rep(2, 13), rep(3, 51),rep(4, 5), rep(5, 9), rep(6, 6),
    rep(1, 12), rep(2, 23), rep(3, 52), rep(4, 11), rep(5, 17), rep(6, 7)))

metadata <- data.frame(  sample = colnames(counts),  condition = condition,  batch = batch)
all(metadata$sample == colnames(counts))

## Step 1 - low-count filtering ##
keep <- rowSums(counts >= 10) >= 22
counts_filtered <- counts[keep, ]
dim(counts_filtered)
summary(rowSums(counts_filtered >= 10))

## Step 2 - Normalization and Transformation##
dds <- DESeqDataSetFromMatrix( countData = counts_filtered, colData = metadata, design = ~ batch + condition)
vsd <- vst(dds, blind = TRUE) # blind kept True as we're not removing batch effect here.
vst_mat <- assay(vsd)

## PCA without batch correction ##
pca <- prcomp(t(vst_mat), scale. = FALSE)
pca_df <- data.frame( PC1 = pca$x[, 1],  PC2 = pca$x[, 2],  condition = metadata$condition,  batch = metadata$batch)
ggplot(pca_df, aes(x = PC1, y = PC2,
                   shape = condition,
                   color = batch)) +
  geom_point(size = 3) +
  theme_classic()
percent_var <- 100 * (pca$sdev^2 / sum(pca$sdev^2))
percent_var[1:5] ## Gets PC 1-5

## Batch correction using limma - removeBatchEffect ##
vst_batch_corrected <- removeBatchEffect(vst_mat, batch = batch)

## PCA after Batch correction ##
pca_after <- prcomp(t(vst_batch_corrected), scale. = FALSE)

pca_df_after <- data.frame( PC1 = pca_after$x[, 1], PC2 = pca_after$x[, 2], Condition = condition, Batch = factor(batch))

plot(
  pca_df_after$PC1,
  pca_df_after$PC2,
  col = as.numeric(pca_df_after$Batch),
  pch = 19,
  xlab = "PC1",
  ylab = "PC2",
  main = "PCA After Batch Correction"
)

## Transpose and export
vst_export <- as.data.frame(t(vst_batch_corrected))
## Set gene names as column names
colnames(vst_export) <- rownames(vst_batch_corrected)
## Add class labels
vst_export$class <- ifelse(condition == "Recovered", 0, 1)
## Add batch labels
vst_export$batch <- as.integer(batch)
vst_export$sample <- rownames(vst_export)
vst_export <- vst_export[, c(
  "sample",
  "class",
  "batch",
  setdiff(colnames(vst_export), c("sample", "class", "batch"))
)]
dim(vst_export)
head(vst_export[, 1:8])
write.table(
  vst_batch_corrected,
  file = "Real_VST_batch_corrected_samples_x_genes_class_batch_090226.tsv",
  sep = "\t",
  quote = FALSE,
  row.names = TRUE,
  col.names = TRUE
)

