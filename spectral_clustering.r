library(PCAmixdata)
library(psych)
library(ggfortify)

count_important_components <- function(eigvals) {
    important_components = vector()
    for (i in 1:length(eigvals))
        if (eigvals[i] > 1)
           important_components = c(important_components, i)
    return(important_components)
}

select_features <- function(specie_scores, important_components, threshold, debug_mode) {
    features = vector()
    for (pc in 1:ncol(specie_scores)){
        if (!pc %in% important_components)
           next
        if (debug_mode)
           print(colnames(specie_scores)[pc])
        for (feat in 1:nrow(specie_scores)){
            if(abs(specie_scores[feat, pc]) > threshold){
                if (debug_mode)
                   print(sprintf("    %s %f",rownames(specie_scores)[feat], specie_scores[feat, pc]))
                if (!rownames(specie_scores)[feat] %in% features)
                    features = c(features, rownames(specie_scores)[feat])
            }
        }
    }
    return(features)
}

# fetches a set
set = read.csv("./tested_datasets/_set-a_4sigmas_complete_fa2_varimax.csv")
rownames(set) = set[,"RecordID"]
set = subset(set, select=-RecordID)

#qualitative_columns = c("Gender", "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4", "MechVent")

## fa1 varimax set a ##
#quantitative_columns = c("NIDiasABP_mean", "NIMAP_mean", "NISysABP_mean", "DiasABP_mean"
#          , "MAP_mean", "AST_mean", "ALT_mean", "HCO3_mean"
#          , "PaCO2_mean", "BUN_mean", "SAPS.I", "SysABP_mean"
#          , "GCS_mean", "HR_mean", "Urine_mean", "K_mean"
#          , "Mg_mean")

## fa2 varimax set a ##
quantitative_columns = c("DiasABP_mean", "MAP_mean", "NIDiasABP_mean", "NIMAP_mean"
        , "AST_mean", "ALT_mean", "HCO3_mean", "PaCO2_mean"
        , "NISysABP_mean", "HR_mean", "K_mean")

## fa1 oblimin set a ##
#quantitative_columns = c("NIDiasABP_mean", "NISysABP_mean", "SysABP_mean", "MAP_mean"
#         , "DiasABP_mean", "HCT_mean", "Albumin_mean", "SOFA"
#         , "Na_mean", "Bilirubin_mean", "HCO3_mean", "Urine_mean")

## fa2 oblimin set a ##
# quantitative_columns = c("Na_mean", "HCT_mean", "Urine_mean", "HCO3_mean")

## FA2 OBLIMIN SET A+B ##
#quantitative_columns = c("SysABP_mean", "HCT_mean", "Temp_mean", "Bilirubin_mean")

## FA2 OBLIMIN SET A+B+C ##
#quantitative_columns = c("Age",	"Albumin_mean",	"HCT_mean", "K_mean", "SysABP_mean")

## complete ##
#quantitative_columns = c('Age','Height','Weight','ALP_mean','ALT_mean','AST_mean'
#                        ,'Albumin_mean','BUN_mean','Bilirubin_mean','Creatinine_mean'
#                        ,'DiasABP_mean','FiO2_mean','GCS_mean','Glucose_mean','HCO3_mean'
#                        ,'HCT_mean','HR_mean','K_mean','Lactate_mean','MAP_mean','Mg_mean'
#                        ,'NIDiasABP_mean','NIMAP_mean','NISysABP_mean','Na_mean'
#                        ,'PaCO2_mean','PaO2_mean','Platelets_mean','SaO2_mean','SysABP_mean'
#                        ,'Temp_mean','Urine_mean','WBC_mean','SAPS.I','SOFA')

## PCA PROCESSING FOR SET A, A+B, A+B+C ##
pca_coords = PCAmix(set[, quantitative_columns], ndim=length(quantitative_columns))
sup = supvar(pca_coords, set[,"outcome"])
important_components_kmeans = count_important_components(pca_coords$eig[,1])
important_components_kmeans_outcome = vector()
for (i in 1:length(important_components_kmeans))
    if (abs(sup$quanti.sup$coord[i]) > 0.1)
        important_components_kmeans_outcome = c(important_components_kmeans_outcome, i)
features = select_features(pca_coords$quanti$coord, important_components_kmeans_outcome, 0.1, FALSE)
# print(subset(pca_coords$ind$coord, select=important_components_kmeans_outcome))
for (i in 1:length(colnames(pca_coords$ind$coord[, important_components_kmeans]))){
    colnames(pca_coords$ind$coord)[i] = sprintf('pc_%i', i)
}

## PCA PROCESSING FOR SETS B AND C ##
#important_dimensions = 1:2
#pca_coords = PCAmix(set[, quantitative_columns], ndim=length(quantitative_columns))
#sup = supvar(pca_coords, set[,"outcome"])
#important_components_kmeans_outcome = vector()
#for (i in important_dimensions)
#    important_components_kmeans_outcome = c(important_components_kmeans_outcome, i)
#print(important_components_kmeans_outcome)
#for (i in 1:length(colnames(pca_coords$ind$coord[, important_dimensions]))){
#    colnames(pca_coords$ind$coord)[i] = sprintf('pc_%i', i)
#}

## write set plus important pca columns ##
#pca_set = set[, features]
#pca_set = cbind(pca_set, set[, qualitative_columns])
#pca_set = cbind(pca_set, pca_coords$ind$coord[, important_components_kmeans])
#pca_set = cbind(pca_set, subset(set,select=outcome))
#set = cbind(set, pca_coords$ind$coord[, important_components_kmeans])
#fix(set)
#write.csv(pca_set, "set_4sigmas_complete_fa2_pca.csv")

## do clustering with alive subset ##
alive_pca_coords = subset(pca_coords$ind$coord, select=important_components_kmeans)[set$outcome == '0', ] ## FOR SET A
#alive_pca_coords = subset(pca_coords$ind$coord, select=important_dimensions)[set$outcome == '0', ] ## FOR SETS B AND C
set.seed(1)
alive_kmeans = kmeans(alive_pca_coords, length(important_components_kmeans), iter.max=300) ## FOR SET A
#alive_kmeans = kmeans(alive_pca_coords, length(important_dimensions), iter.max=300) ## FOR SETS B AND C
set[set$outcome == '0', 'kmeans_labels'] = alive_kmeans$cluster

## do clustering with dead subset ##
dead_pca_coords = subset(pca_coords$ind$coord, select=important_components_kmeans)[set$outcome == '1', ] ## FOR SET A
#dead_pca_coords = subset(pca_coords$ind$coord, select=important_dimensions)[set$outcome == '1', ] ## FOR SETS B AND C
set.seed(1)
dead_kmeans = kmeans(dead_pca_coords, length(important_components_kmeans), iter.max=300) ## FOR SET A
#dead_kmeans = kmeans(dead_pca_coords, length(important_dimensions), iter.max=300) ## FOR SETS B AND C
set[set$outcome == '1', 'kmeans_labels'] = dead_kmeans$cluster+length(important_components_kmeans) ## FOR SET A
#set[set$outcome == '1', 'kmeans_labels'] = dead_kmeans$cluster+length(important_dimensions) ## FOR SETS B AND C

## write set plus kmeans labels and pca columns
kmeans_set = set[, features] ## FOR SET A
# kmeans_set = set ## FOR SETS B AND C
# kmeans_set = cbind(kmeans_set, set[, quantitative_columns]) ## FOR SET A
kmeans_set = cbind(kmeans_set, pca_coords$ind$coord[, important_components_kmeans])  ## FOR SET A
#kmeans_set = cbind(kmeans_set, pca_coords$ind$coord[, important_dimensions]) ## FOR SETS B AND C
kmeans_set = cbind(kmeans_set, subset(set,select=c(outcome, kmeans_labels))) ## FOR SET A
write.csv(kmeans_set, "./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans.csv")
#print(dead_kmeans)