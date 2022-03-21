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
set = read.csv("./tested_datasets/set-a_4sigmas_complete.csv")
#print(colnames(set))
quantitative_columns = c('Age','Height','Weight','ALP_mean','ALT_mean','AST_mean'
			,'Albumin_mean','BUN_mean','Bilirubin_mean','Creatinine_mean'
			,'DiasABP_mean','FiO2_mean','GCS_mean','Glucose_mean','HCO3_mean'
			,'HCT_mean','HR_mean','K_mean','Lactate_mean','MAP_mean','Mg_mean'
			,'NIDiasABP_mean','NIMAP_mean','NISysABP_mean','Na_mean'
			,'PaCO2_mean','PaO2_mean','Platelets_mean','SaO2_mean','SysABP_mean'
			,'Temp_mean','Urine_mean','WBC_mean','SAPS.I','SOFA')

#pca with PCAmixdata
pca = PCAmix(set[,quantitative_columns], ndim=length(quantitative_columns))
sup = supvar(pca, set[,"outcome"])
#for (i in 1:length(quantitative_columns))
#    for (j in i:length(quantitative_columns))
#        if(j > i)
#	     plot(sup, choice="cor", axes=c(i,j))

#print(sup$quanti.sup$coord)
important_components = count_important_components(pca$eig[,1])
important_components_outcome = vector()
for (i in 1:length(important_components))
    if (abs(sup$quanti.sup$coord[i]) > 0.1)
        important_components_outcome = c(important_components_outcome, i)
#features = select_features(pca$quanti$coord, important_components, 0.2, TRUE)
features = select_features(pca$quanti$coord, important_components_outcome, 0.1, FALSE)

#cat(paste(cat(gsub("dim", "PC", colnames(pca$quanti$coord[,important_components])), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Eigvals.", round(pca$eig[important_components], digits=2), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Outcome", round(sup$quanti.sup$coord[, important_components], digits=2), sep=" & "), "\\\\", "\n"))
#for (rowname in rownames(pca$quanti$coord)){
#    cat(paste(cat(unlist(strsplit(rowname, split="_mean"))[1], round(pca$quanti$coord[rowname,important_components], digits=2), sep=" & "), "\\\\", "\n"))
#}

print("PCA")
print(features)

### VARIMAX ###
#factor analysis with psych
fa_1 = principal(set[,features], nfactors=length(features), rotate="varimax", scores=TRUE)
fa_1_components = count_important_components(fa_1$values)
fa_1_features = select_features(fa_1$loadings, fa_1_components, 0.7, FALSE)

#cat(paste(cat(gsub("RC", "F", colnames(fa_1$loadings[,fa_1_components])), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Eigvals.", round(fa_1$values[fa_1_components], digits=2), sep=" & "), "\\\\", "\n"))
#for (rowname in rownames(fa_1$loadings[,fa_1_components])){
#    cat(paste(cat(unlist(strsplit(rowname, split="_mean"))[1], round(fa_1$loadings[rowname,fa_1_components], digits=2), sep=" & "), "\\\\", "\n"))
#}

print("FA_1 - VARIMAX")
print(fa_1_features)

#factor analysis with psych
fa_2 = principal(set[,fa_1_features], nfactors=length(fa_1_features), rotate="varimax", scores=TRUE)
fa_2_components = count_important_components(fa_2$values)
fa_2_features = select_features(fa_2$loadings, fa_2_components, 0.7, FALSE)

#cat(paste(cat(gsub("RC", "F", colnames(fa_2$loadings[,fa_2_components])), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Eigvals.", round(fa_2$values[fa_2_components], digits=2), sep=" & "), "\\\\", "\n"))
#for (rowname in rownames(fa_2$loadings[,fa_2_components])){
#    cat(paste(cat(unlist(strsplit(rowname, split="_mean"))[1], round(fa_2$loadings[rowname,fa_2_components], digits=2), sep=" & "), "\\\\", "\n"))
#}

print("FA_2 - VARIMAX")
print(fa_2_features)

### OBLIMIN ###
#factor analysis with psych
fa_1 = principal(set[,features], nfactors=length(features), rotate="oblimin", scores=TRUE)
fa_1_components = count_important_components(fa_1$values)
fa_1_features = select_features(fa_1$loadings, fa_1_components, 0.7, FALSE)

#cat(paste(cat(gsub("TC", "F", colnames(fa_1$loadings[,fa_1_components])), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Eigvals.", round(fa_1$values[fa_1_components], digits=2), sep=" & "), "\\\\", "\n"))
#for (rowname in rownames(fa_1$loadings[,fa_1_components])){
#    cat(paste(cat(unlist(strsplit(rowname, split="_mean"))[1], round(fa_1$loadings[rowname,fa_1_components], digits=2), sep=" & "), "\\\\", "\n"))
#}

print("FA_1 - OBLIMIN")
print(fa_1_features)

#factor analysis with psych
fa_2 = principal(set[,fa_1_features], nfactors=length(fa_1_features), rotate="oblimin", scores=TRUE)
fa_2_components = count_important_components(fa_2$values)
fa_2_features = select_features(fa_2$loadings, fa_2_components, 0.7, FALSE)

#cat(paste(cat(gsub("TC", "F", colnames(fa_2$loadings[,fa_2_components])), sep=" & "), "\\\\", "\n"))
#cat(paste(cat("Eigvals.", round(fa_2$values[fa_2_components], digits=2), sep=" & "), "\\\\", "\n"))
#for (rowname in rownames(fa_2$loadings[,fa_2_components])){
#    cat(paste(cat(unlist(strsplit(rowname, split="_mean"))[1], round(fa_2$loadings[rowname,fa_2_components], digits=2), sep=" & "), "\\\\", "\n"))
#}

print("FA_2 - OBLIMIN")
print(fa_2_features)