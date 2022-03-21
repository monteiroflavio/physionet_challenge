library(PCAmixdata)

# fetch set
set = read.csv("./tested_datasets/set-a+b+c_4_sigmas_complete.csv")
# remove record id column
# create set with time series quantitative variables
set.quant = subset(set, select=-c(RecordID,Age,Height,Weight,SAPS.I,SOFA,Gender,ICUType_1,ICUType_2,ICUType_3,ICUType_4,MechVent,outcome))

for (i in seq(from=1, to=ncol(set.quant), by=3)){
    pca = PCAmix(set.quant[,i:(i+2)], ndim=ncol(set.quant[,i:(i+2)]))
    sup = supvar(pca, set[,ncol(set)])
    plot(sup, choice="cor", axes=c(1,2))
    plot(sup, choice="cor", axes=c(1,3))
    plot(sup, choice="cor", axes=c(2,3))
}