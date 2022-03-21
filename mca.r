library(PCAmixdata)

# fetches a set
set = read.csv("./tested_datasets/set-a_categorical.csv", header=TRUE, sep=",")

# create set with qualitative variables
set.quali = subset(set, select=c(Gender,MechVent,ICUType_1,ICUType_2,ICUType_3,ICUType_4))

# append outcome to final qualitative set
set.quali$outcome = set$outcome

# stringify values
set.quali[] <- lapply(set.quali, as.character)

# rename columns for dissertation, old names order: "Gender", "MechVent", "ICUType_1", "ICUType_2", "ICUType_3", "ICUType_4", "outcome"
colnames(set.quali) = c("a", "b", "c", "d", "e", "f", "y")

# correspondence analysis
mca = PCAmix(X.quali = set.quali, ndim=ncol(set.quali), rename.level=TRUE)
#sup = supvar(mca, set$outcome, rename.level=TRUE)

for (i in 1:(length(mca$eig)/3))
    for (j in i:(length(mca$eig)/3))
    	if(j > i)
    	     plot(mca, choice="levels", axes=c(i,j))

