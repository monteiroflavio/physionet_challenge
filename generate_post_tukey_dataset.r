library(agricolae)
source("slice_dataset.r")

set.kmeans <- read.csv(file="./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN+GCS+Na.csv", header=TRUE, sep=",")
rownames(set.kmeans) = set.kmeans[,"RecordID"]
set.kmeans = subset(set.kmeans, select=-RecordID)

set.full = read.csv(file="./tested_datasets/set-a_4sigmas.csv", header=TRUE, sep=",")
rownames(set.full) = set.full[,"RecordID"]
set.full = subset(set.full, select=-c(RecordID,outcome))

## fa2 varimax set a ##
# columns = c("Albumin_mean", "GCS_mean", "Age", "BUN_mean", "Na_mean")

## fa2 oblimin set a ##
#columns = c("Albumin_mean", "SysABP_mean", "BUN_mean", "GCS_mean")#, "AST_mean", "ALP_mean"
#columns = c("BUN_mean", "GCS_mean")#, "Na_mean"

## fa2 oblimin set a+b ##
#columns = c("GCS_mean", "SAPS.I", "Age", "BUN_mean")#, "SaO2_mean", "SOFA", "PaO2_mean"

## set a categorical ##
columns = c("ICUType", "Gender", "MechVent")

## set a categorical ##
#columns = c("ICUType", "MechVent")#, "Gender"

## fa2 oblimin set a+b+c ##
#columns = c("GCS_mean", "FiO2_mean", "SOFA")

## individual variable's tests set a ##
#columns = c("BUN_mean", "Age", "Temp_mean", "Urine_mean", "GCS_mean", "SysABP_mean")#, "HCO3_mean")

set.column = slice_dataset(set.full
                           , z_scale=TRUE
                           , first_cut_threshold=0.7
                           , columns=columns
                           , drop_na_first=FALSE)

set = cbind(set.kmeans[intersect(rownames(set.kmeans),rownames(set.column)),]
            , set.column[intersect(rownames(set.kmeans),rownames(set.column)),])

write.csv(set, "./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans+BUN+GCS+Na+cat.csv")