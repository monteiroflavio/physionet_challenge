library(agricolae)
library(ggplot2)
source("slice_dataset.r")

set.full = read.csv(file="./tested_datasets/set-a_4sigmas.csv", header=TRUE, sep=",")
rownames(set.full) = set.full[,"RecordID"]
# set.full = subset(set.full, select=-c(RecordID,Gender,MechVent,outcome))
set.full = subset(set.full, select=-c(RecordID,Gender,ICUType,MechVent))

set.kmeans <- read.csv(file="./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans.csv", header=TRUE, sep=",")
rownames(set.kmeans) = set.kmeans[,"RecordID"]
set.kmeans = subset(set.kmeans, select=-RecordID)

set.complete = read.csv(file="./tested_datasets/set-a_4sigmas_complete_pca.csv", header=TRUE, sep=",")
set.complete = subset(set.complete, select=-RecordID)
# set.complete = subset(set.complete, select=-c(Gender,MechVent,ICUType_1,ICUType_2,ICUType_3,ICUType_4,outcome))

tukey_columns = setdiff(colnames(set.complete), colnames(set.kmeans))

for (i in 1:length(tukey_columns)){
    print(tukey_columns[i])
    set.column = slice_dataset(set.full
				, z_scale=TRUE
				#, first_cut_threshold=0.7
				, dummy_columns=c()
				, columns=c(tukey_columns[i], "outcome")
				, drop_na_first=FALSE)
    #fix(set.column)
    set = cbind(set.kmeans[intersect(rownames(set.kmeans),rownames(set.column)),]
    		, set.column[intersect(rownames(set.kmeans),rownames(set.column)),tukey_columns[i]])

    colnames(set)[ncol(set)] = tukey_columns[i]
    tukey_result = HSD.test(do.call("aov", list(formula = as.formula(paste(tukey_columns[i],'~kmeans_labels')), data = quote(set))),trt="kmeans_labels", group=TRUE)
    # tukey_result = HSD.test(do.call("aov", list(formula = as.formula(paste(tukey_columns[i],'~outcome')), data = quote(set.column))),trt="outcome", group=TRUE)
    # print(tukey_result$group)
    if (length(unique(tukey_result$group[,2])) > 1){
       #do.call("boxplot", list(formula=as.formula(paste(tukey_columns[i],'~kmeans_labels')), data=quote(set), main=tukey_columns[i], range=2, outline=FALSE))
       #plot(tukey_result, main=tukey_columns[i])
       write.csv(set, sprintf("./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans+%s.csv", tukey_columns[i]))
    #    write.csv(set.column, sprintf("./tested_datasets/_set-a_4sigmas_complete_fa2_varimax_kmeans+%s.csv", tukey_columns[i]))
    }
}