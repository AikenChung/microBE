#===============================================================================
# HeapMap visualization for spotting the batch effects of Phyla dataset
# Team: microBE
# @Ali, @Anthony, @Laura, @Aiken
#===============================================================================
library(mixOmics) # for CLR
library(pheatmap) # for heatmap

# Prepare input data and get data ready
phyla_dataset_d3 <- read.csv("/file_Path_To/phyla_dataset_d3.csv",
                             row.names=1,
                             header=T,
                             check.names=FALSE)
phyla.count <- data.matrix(phyla_dataset_d3[,1:1177], rownames.force = NA)
phyla.metadata <- phyla_dataset_d3[,1178:1183]

#************************** Setting Parameters *********************************#
# Set parameters
# The threshold 0.01 denotes the frequency of 0.01 %
# frequency_filtering_threshold <- 0.01 # Frequency is a bit of indirect! Using the count number.
totalCount_threshold <- 5
# Color codes for heatmap visualization
batch_color_col_site <- c('OSCCAR' = 'lightsalmon2', 
                          'PRISM' = 'blue',
                          'RISK' = 'yellow', 
                          'MGH' = 'lightslateblue',
                          'Cedars-Sinai' = 'limegreen', 
                          'Cincinnati' = 'magenta2',
                          'MGH Pediatrics' = 'mediumorchid4', 
                          'BOULDER' = 'navajowhite2',
                          'North Carolina' = 'olivedrab', 
                          'CVDF' = 'red',
                          'paneth_cells' = 'green1', 
                          'PRJNA436359' = 'darkturquoise',
                          'AG' = 'darkgreen')
batch_color_stool_biopsy <- c('stool' = 'darkgrey', 'biopsy' = 'darkred')
batch_color_studyID <- c('GEVERSM' = 'yellow', 
                         'HMP' = 'lightslateblue',
                         'GEVERSC' = 'limegreen', 
                         'QIITA550' = 'magenta2',
                         'QIITA2202' = 'mediumorchid4', 
                         'CVDF' = 'navajowhite2',
                         'MUC' = 'olivedrab', 
                         'PRJNA436359' = 'red',
                         'AG' = 'green1')
group_color_uc_cd <- c('Control' = 'springgreen', 
                       'CD' = 'dodgerblue3', 
                       'UC' = 'deeppink')
# Set color code
batch_color_setting <- batch_color_col_site # or batch_color_stool_biopsy, batch_color_studyID 
group_color_setting <- group_color_uc_cd # or batch_color_col_site, batch_color_stool_biopsy, batch_color_studyID 
# Set batch effect variable and the observing group
batch_effect_variable <- phyla.metadata$col_site # $uc_cd, $studyID, $col_site, $stool_biopsy
observing_group <- phyla.metadata$uc_cd  # $uc_cd, $studyID, $col_site, $stool_biopsy
# Set title for heatmap graph
rawCount_heatMapt_Title <- 'Phyla dataset - Raw Count'
normalized_heatMapt_Title <- 'Phyla dataset - Normalized'

#*************************** End of Setting Parameters **************************#

# Extract batch effect variables and observing group as factors
phyla.batch <- as.factor(batch_effect_variable)
names(phyla.batch) <- row.names(phyla.metadata)
phyla.group <- as.factor(observing_group)
names(phyla.group) <- row.names(phyla.metadata)

# Prefiltering basing on the frequency of the genus
#phyla.index.keep <- which(colSums(phyla.count)*100/(sum(colSums(phyla.count))) > frequency_filtering_threshold)
phyla.index.keep <- which((colSums(phyla.count)) > totalCount_threshold)
phyla.count.keep.ori <- phyla.count[, phyla.index.keep]
# Re-organize the genus name (because it's too long to visualize)
extracted_new_colName <- colnames(phyla.count.keep.ori)
i<-1
for (name in extracted_new_colName){
  name_list <- list()
  name_list <- strsplit(name, ";", fixed = TRUE)
  firstName = name_list[[1]][1]
  lastName = ""
  name_list_Rev <- rev(name_list[[1]])
  for(elementName in name_list_Rev){
     if(elementName != "__" &  nchar(elementName)>9){
       lastName <- elementName
       break;
     }
  }
  new_colName <-paste(firstName, lastName, sep = ";", collapse = NULL)
  extracted_new_colName[i] <- new_colName
  i <- i+1
}
phyla.count.keep <- phyla.count.keep.ori
colnames(phyla.count.keep) <- extracted_new_colName
dim(phyla.count.keep)

# Adding offset to filtered raw count data
phyla.count.keep <- phyla.count.keep + 1

# Centered log-ratio transformation
phyla.clr <- logratio.transfo(phyla.count.keep, logratio = 'CLR')
class(phyla.clr) <- 'matrix'

# For Batch Effect detection
# Function to draw Heatmap
drawHeatMap <- function(inputData, annoCol, annoMetaColor, titleText){
  pheatmap(inputData, 
           scale = 'none', 
           cluster_rows = F, 
           cluster_cols = T, 
           fontsize_row = 4, fontsize_col = 6,
           fontsize = 8,
           clustering_distance_rows = 'euclidean',
           clustering_method = 'ward.D',
           treeheight_row = 30,
           annotation_col = annoCol,
           annotation_colors = annoMetaColor,
           border_color = 'NA',
           main = titleText)
}


# Filtered Raw Count Data for drawHeatMap function
phyla.count.keep.scale <- scale(phyla.count.keep,center = T, scale = T)
phyla.count.keep.scale <- scale(t(phyla.count.keep.scale), center = T, scale = T)
# Filtered Normalized Data for drawHeatMap function
phyla.clr.scale <- scale(phyla.clr,center = T, scale = T)
phyla.clr.scale <- scale(t(phyla.clr.scale), center = T, scale = T)
# Set attributes for drawHeatMap function
phyla.anno_col <- data.frame(Batch = phyla.batch, Group = phyla.group)
phyla.anno_metabo_colors <- list(Batch = batch_color_setting, 
                                 Group = group_color_setting)

# Darw the heatmap for Filtered Raw Count Data
drawHeatMap(phyla.count.keep.scale, phyla.anno_col,phyla.anno_metabo_colors,rawCount_heatMapt_Title)
# Darw the heatmap for Filtered Normalized Data
drawHeatMap(phyla.clr.scale, phyla.anno_col,phyla.anno_metabo_colors,normalized_heatMapt_Title)
