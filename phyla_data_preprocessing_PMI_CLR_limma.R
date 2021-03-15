#===============================================================================
# Data preprocessing with PMI (pointwise mutual information) filtering and 
# CLR (centered log-ratio) normalization for phyla microbiotome count data
# The PMI filtered file and PMI filtered with limma BE removal file will be generated.
# 
# Team: microBE
# @Ali, @Anthony, @Laura, @Aiken
#===============================================================================

library(limma) # removeBatchEffect (LIMMA)
library(mixOmics) # CLR

phyla_dataset_d3 <- read.csv("/file_Path_To/phyla_dataset_d3.csv", 
                             row.names=1,
                             header=T,
                             check.names=FALSE)

# Please update the file name to save here:
noBE_removal_fileName <- "phyla_all_5267x1177_PMI_threshold_0_clr.csv"
BE_removal_fileName <- "phyla_all_5267x1177_PMI_threshold_0_clr_limma.csv"

# Please change the data contenet here:
## Only stool samples
# phyla_dataset_d3 <- phyla_dataset_d3[ grepl( "stool" , phyla_dataset_d3$stool_biopsy ), ]
## Only biopsy samples
# phyla_dataset_d3 <- phyla_dataset_d3[ grepl( "biopsy" , phyla_dataset_d3$stool_biopsy ), ]
## remove "Cedars-Sinai"
# phyla_dataset_d3 <- phyla_dataset_d3[ !grepl( "Cedars-Sinai" , phyla_dataset_d3$col_site ), ]
## remove "North Carolina"
# phyla_dataset_d3 <- phyla_dataset_d3[ !grepl( "North Carolina" , phyla_dataset_d3$col_site ), ]

# data obj
phyla.count <- data.matrix(phyla_dataset_d3[,1:1177], rownames.force = NA)

# metadata obj
phyla.metadata <- phyla_dataset_d3[,1178:1183]
names(phyla.count) <- row.names(phyla.metadata)

# Extract batch effect variables and observing group as factors
phyla.batch <- as.factor(phyla.metadata$col_site)
phyla.group <- as.factor(phyla.metadata$uc_cd)
names(phyla.group) <- row.names(phyla.count)


# Positive PMI is used here, so set threshold value to zero
pmi_threshold <- 0 

# function for calculating. Thanks to Ahmad!!
pmi <- function(matrix, smooth_val=1){
  matrix <- matrix + smooth_val
  matrix <- matrix / sum(matrix)
  sumOfRow <- rowSums(matrix)
  sumOfCol <- colSums(matrix)
  rowDiv <- sweep(matrix, 1, sumOfRow, "/")
  colDiv <- sweep(rowDiv, 2, sumOfCol, "/")
  log.transform.mat <- log(colDiv,10)
  return (log.transform.mat)
}

# PMI Calculation
class(phyla.count) <- 'matrix'
phyla_pmi_transform <- pmi(phyla.count, 1)
# PMI filtering with threshold value set to zero
threshold_vec <- matrix(pmi_threshold,dim(phyla_pmi_transform)[1],1)
# filtered out --> 0, preserved --> 1
filtered.phyla.pmi.mat <- t(apply(cbind(threshold_vec ,phyla_pmi_transform), 1, function(x) {
  threshold_vec=x[1]
  x=x[-1]
  ifelse(x>threshold_vec, 1, ifelse(x<=threshold_vec, 0, NA))
}))

# filtered out count data will be set to zero
phyla.count.keep.all <- phyla.count*filtered.phyla.pmi.mat

# For comparison sake
phyla.count.keep <- phyla.count.keep.all
colnames(phyla.count.keep) <- colnames(phyla.count.keep.all)
dim(phyla.count.keep)

# Adding offset to filtered raw count data
phyla.count.keep <- phyla.count.keep + 1

# Centered log-ratio transformation
phyla.clr <- logratio.transfo(phyla.count.keep, logratio = 'CLR')
class(phyla.clr) <- 'matrix'

#========================= Batch Effect Correction ===========================
# limma package, removeBatchEffect method for BE correction
phyla.mod <- model.matrix( ~ phyla.group)
phyla.limma <- t(removeBatchEffect(t(phyla.clr), batch = phyla.batch, 
                                   design = phyla.mod))

#========================= End of Batch Effect Correction=====================


# Saving the filtered CLR data (before and After BE)
phyla.clr.round <- round(phyla.clr[,1:1177], digits = 4)
phyla.limma.round <- round(phyla.limma[,1:1177], digits = 4)
phyla.clr.round <- cbind(phyla.clr.round,phyla.metadata)
phyla.limma.round <- cbind(phyla.limma.round,phyla.metadata)
write.csv(phyla.clr.round,noBE_removal_fileName, row.names = TRUE)
write.csv(phyla.limma.round,BE_removal_fileName, row.names = TRUE)


