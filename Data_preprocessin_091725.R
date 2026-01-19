## Author - John George - 091725
## Data Preprocessing - 177 COVID-19 vs 171 LC


setwd("Path to directory")
library(DESeq2)
library(ggplot2)
library(ggrepel)

counts <- read.table("177COV-171LC_raw-counts.tsv", header = TRUE, row.names = 1)
dim(counts)

condition <- factor(c(rep("COVID-19", 177), rep("Long_COVID", 171)))
coldata <- data.frame(row.names = colnames(counts), condition)

# Estimate size Factor - normalization
dds <- DESeqDataSetFromMatrix(countData = counts,
                              colData = coldata,
                              design = ~ condition)

dds <- estimateSizeFactors(dds)

#Variance Stabilizing Transformation
vsd <- vst(dds, blind = TRUE)
vst_mat <- assay(vsd)
dim(vst_mat)
write.table(vst_mat, "177cov171LC-HIC-ISG_EST_VST_103025.tsv", sep = "\t", quote = FALSE) 

df_vst_mat <- read.table("177cov171LC_EST_VST_102325.tsv", sep='\t', header = TRUE) 
colnames(df_vst_mat)
gene_ensemble <- read.csv("gene-HIC-ISG_symbol_to_ensembl_103025.csv", header = TRUE)
head(gene_ensemble)

#Plot PCA
plotPCA(vsd, intgroup = "condition" , ntop = 787)
pcaData <- plotPCA(vsd, intgroup = "condition", returnData = TRUE)
pcaData$Sample <- rownames(pcaData)

ggplot(pcaData, aes(PC1, PC2, color = condition, label = Sample)) +
  geom_point(size = 3) +
  geom_text_repel(size = 3) +  
  xlab(paste0("PC1: ")) +
  ylab(paste0("PC2: ")) +
  coord_fixed() +
  theme_classic()
  
## Transpose and add Class labels
vst_export <- as.data.frame(t(hgnc_counts[,-1]))# samples × genes
colnames(vst_export) <- hgnc_counts$GeneSymbol       # set column names
colnames(vst_export)
vst_export$class_label <- ifelse(condition == "COVID-19", 0, 1)
write.table(vst_export, "177cov171LC_EST-VST_103025.tsv", sep = "\t", quote = FALSE, )






