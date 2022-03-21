library(dummies)

slice_dataset <- function(data
        , quantitative_columns = c('Age', 'Height', 'Weight', 'SOFA', 'SAPS.I'
            , 'Albumin_1', 'Albumin_2', 'Albumin_mean'
            , 'ALP_1', 'ALP_2', 'ALP_mean'
            , 'ALT_1', 'ALT_2', 'ALT_mean'
            , 'AST_1', 'AST_2', 'AST_mean'
            , 'Bilirubin_1', 'Bilirubin_2', 'Bilirubin_mean'
            , 'BUN_1', 'BUN_2', 'BUN_mean'
            , 'Cholesterol_1', 'Cholesterol_2', 'Cholesterol_mean'
            , 'Creatinine_1', 'Creatinine_2', 'Creatinine_mean'
            , 'FiO2_1', 'FiO2_2', 'FiO2_mean'
            , 'GCS_1', 'GCS_2', 'GCS_mean'
            , 'Glucose_1', 'Glucose_2', 'Glucose_mean'
            , 'HCO3_1', 'HCO3_2', 'HCO3_mean'
            , 'HCT_1', 'HCT_2', 'HCT_mean'
            , 'HR_1', 'HR_2', 'HR_mean'
            , 'K_1', 'K_2', 'K_mean'
            , 'Lactate_1', 'Lactate_2', 'Lactate_mean'
            , 'Mg_1', 'Mg_2', 'Mg_mean'
            , 'Na_1', 'Na_2', 'Na_mean'
            , 'PaCO2_1', 'PaCO2_2', 'PaCO2_mean'
            , 'PaO2_1', 'PaO2_2', 'PaO2_mean'
            , 'pH_1', 'pH_2', 'pH_mean'
            , 'Platelets_1', 'Platelets_2', 'Platelets_mean'
            , 'RespRate_1', 'RespRate_2', 'RespRate_mean'
            , 'SaO2_1', 'SaO2_2', 'SaO2_mean'
            , 'Temp_1', 'Temp_2', 'Temp_mean'
            , 'TroponinI_1', 'TroponinI_2', 'TroponinI_mean'
            , 'TroponinT_1', 'TroponinT_2', 'TroponinT_mean'
            , 'Urine_1', 'Urine_2', 'Urine_mean'
            , 'WBC_1', 'WBC_2', 'WBC_mean'
            , 'DiasABP_1', 'DiasABP_2', 'DiasABP_mean'
            , 'NIDiasABP_1', 'NIDiasABP_2', 'NIDiasABP_mean'
            , 'SysABP_1', 'SysABP_2', 'SysABP_mean'
            , 'NISysABP_1', 'NISysABP_2', 'NISysABP_mean'
            , 'MAP_1', 'MAP_2', 'MAP_mean'
            , 'NIMAP_1', 'NIMAP_2', 'NIMAP_mean')
        , dummy_columns = c('ICUType')
        , columns = c('Age', 'Height', 'Weight', 'SOFA', 'SAPS.I'
            , 'Albumin_1', 'Albumin_2', 'Albumin_mean'
            , 'ALP_1', 'ALP_2', 'ALP_mean'
            , 'ALT_1', 'ALT_2', 'ALT_mean'
            , 'AST_1', 'AST_2', 'AST_mean'
            , 'Bilirubin_1', 'Bilirubin_2', 'Bilirubin_mean'
            , 'BUN_1', 'BUN_2', 'BUN_mean'
            , 'Cholesterol_1', 'Cholesterol_2', 'Cholesterol_mean'
            , 'Creatinine_1', 'Creatinine_2', 'Creatinine_mean'
            , 'FiO2_1', 'FiO2_2', 'FiO2_mean'
            , 'GCS_1', 'GCS_2', 'GCS_mean'
            , 'Glucose_1', 'Glucose_2', 'Glucose_mean'
            , 'HCO3_1', 'HCO3_2', 'HCO3_mean'
            , 'HCT_1', 'HCT_2', 'HCT_mean'
            , 'HR_1', 'HR_2', 'HR_mean'
            , 'K_1', 'K_2', 'K_mean'
            , 'Lactate_1', 'Lactate_2', 'Lactate_mean'
            , 'Mg_1', 'Mg_2', 'Mg_mean'
            , 'Na_1', 'Na_2', 'Na_mean'
            , 'PaCO2_1', 'PaCO2_2', 'PaCO2_mean'
            , 'PaO2_1', 'PaO2_2', 'PaO2_mean'
            , 'pH_1', 'pH_2', 'pH_mean'
            , 'Platelets_1', 'Platelets_2', 'Platelets_mean'
            , 'RespRate_1', 'RespRate_2', 'RespRate_mean'
            , 'SaO2_1', 'SaO2_2', 'SaO2_mean'
            , 'Temp_1', 'Temp_2', 'Temp_mean'
            , 'TroponinI_1', 'TroponinI_2', 'TroponinI_mean'
            , 'TroponinT_1', 'TroponinT_2', 'TroponinT_mean'
            , 'Urine_1', 'Urine_2', 'Urine_mean'
            , 'WBC_1', 'WBC_2', 'WBC_mean'
            , 'DiasABP_1', 'DiasABP_2', 'DiasABP_mean'
            , 'NIDiasABP_1', 'NIDiasABP_2', 'NIDiasABP_mean'
            , 'SysABP_1', 'SysABP_2', 'SysABP_mean'
            , 'NISysABP_1', 'NISysABP_2', 'NISysABP_mean'
            , 'MAP_1', 'MAP_2', 'MAP_mean'
            , 'NIMAP_1', 'NIMAP_2', 'NIMAP_mean', 'Gender'
            , 'MechVent', 'ICUType', 'outcome')
        , z_scale = TRUE
        , log_columns = c('Weight', 'Albumin_1', 'Albumin_2', 'Albumin_mean'
      , 'ALP_1', 'ALP_2', 'ALP_mean'
      , 'ALT_1', 'ALT_2', 'ALT_mean'
      , 'AST_1', 'AST_2', 'AST_mean'
      , 'Bilirubin_1', 'Bilirubin_2', 'Bilirubin_mean'
      , 'BUN_1', 'BUN_2', 'BUN_mean'
      , 'Cholesterol_1', 'Cholesterol_2', 'Cholesterol_mean'
      , 'Creatinine_1', 'Creatinine_2', 'Creatinine_mean'
      , 'FiO2_1', 'FiO2_2', 'FiO2_mean'
      , 'Glucose_1', 'Glucose_2', 'Glucose_mean'
      , 'HCO3_1', 'HCO3_2', 'HCO3_mean'
      , 'HCT_1', 'HCT_2', 'HCT_mean'
      , 'HR_1', 'HR_2', 'HR_mean'
      , 'K_1', 'K_2', 'K_mean'
      , 'Lactate_1', 'Lactate_2', 'Lactate_mean'
      , 'Mg_1', 'Mg_2', 'Mg_mean'
      , 'Na_1', 'Na_2', 'Na_mean'
      , 'PaCO2_1', 'PaCO2_2', 'PaCO2_mean'
      , 'PaO2_1', 'PaO2_2', 'PaO2_mean'
      , 'pH_1', 'pH_2', 'pH_mean'
      , 'Platelets_1', 'Platelets_2', 'Platelets_mean'
      , 'RespRate_1', 'RespRate_2', 'RespRate_mean'
      , 'Temp_1', 'Temp_2', 'Temp_mean'
      , 'TroponinI_1', 'TroponinI_2', 'TroponinI_mean'
      , 'TroponinT_1', 'TroponinT_2', 'TroponinT_mean'
      , 'Urine_1', 'Urine_2', 'Urine_mean'
      , 'WBC_1', 'WBC_2', 'WBC_mean'
      , 'DiasABP_1', 'DiasABP_2', 'DiasABP_mean'
      , 'NIDiasABP_1', 'NIDiasABP_2', 'NIDiasABP_mean'
      , 'SysABP_1', 'SysABP_2', 'SysABP_mean'
      , 'NISysABP_1', 'NISysABP_2', 'NISysABP_mean'
      , 'MAP_1', 'MAP_2', 'MAP_mean'
      , 'NIMAP_1', 'NIMAP_2', 'NIMAP_mean')
        , first_cut_threshold = 0
        , remove_time_series = FALSE
        , drop_na_first = FALSE
        ) {
  if (first_cut_threshold > 0) {
    data = data[, sapply(data, function(x) sum(is.na(x) / length(x))) <= first_cut_threshold]
    columns = intersect(colnames(data), columns)
    quantitative_columns = intersect(colnames(data), quantitative_columns)
    log_columns = intersect(colnames(data), log_columns)
  }
  if (drop_na_first)
    data = data[complete.cases(data),]
  if (length(dummy_columns) > 0) {
    for (dummie in dummy_columns) {
      data = cbind(data, dummy(data[, dummie], sep = "_"))
      data = data[, - grep(dummie, colnames(data))]
      colnames(data) = sub("set_", sprintf("%s_", dummie), colnames(data))
      if (length(columns) > 0 && dummie %in% columns) {
        columns = columns[-grep(dummie, columns)]
        columns = c(columns, colnames(data)[grep(dummie, colnames(data))])
      }
    }
  }
  if (length(log_columns) > 0)
    data[, log_columns] <- log(data[, log_columns] + 1)
  if (z_scale)
    data[, quantitative_columns] <- scale(data[, quantitative_columns])
  if (length(columns) > 0)
    data = data[, columns]
  if (!drop_na_first) {
    if (length(columns) == 1)
      data = data[complete.cases(data)]
    else
      data = data[complete.cases(data),]
  }
  return(data)
}

# fetches a set
set = read.csv("./tested_datasets/set-c_4sigmas.csv")
rownames(set) = set[, "RecordID"]
set = subset(set, select = -RecordID)

# #print(dim(result))

## SET A COMPLETE AFTER DATA CLEANING ##
# columns = c('Age', 'Height', 'Weight', 'ALP_1', 'ALP_2'
#             , 'ALP_mean', 'ALT_1', 'ALT_2', 'ALT_mean'
#             , 'AST_1', 'AST_2', 'AST_mean', 'Albumin_1'
#             , 'Albumin_2', 'Albumin_mean', 'BUN_1', 'BUN_2'
#             , 'BUN_mean', 'Bilirubin_1', 'Bilirubin_2', 'Bilirubin_mean'
#             , 'Creatinine_1', 'Creatinine_2', 'Creatinine_mean'
#             , 'DiasABP_1', 'DiasABP_2', 'DiasABP_mean', 'FiO2_1'
#             , 'FiO2_2', 'FiO2_mean', 'GCS_1', 'GCS_2', 'GCS_mean'
#             , 'Glucose_1', 'Glucose_2', 'Glucose_mean', 'HCO3_1'
#             , 'HCO3_2', 'HCO3_mean', 'HCT_1', 'HCT_2', 'HCT_mean'
#             , 'HR_1', 'HR_2', 'HR_mean', 'K_1', 'K_2', 'K_mean'
#             , 'Lactate_1', 'Lactate_2', 'Lactate_mean', 'MAP_1'
#             , 'MAP_2', 'MAP_mean', 'Mg_1', 'Mg_2', 'Mg_mean'
#             , 'NIDiasABP_1', 'NIDiasABP_2', 'NIDiasABP_mean'
#             , 'NIMAP_1', 'NIMAP_2', 'NIMAP_mean', 'NISysABP_1'
#             , 'NISysABP_2', 'NISysABP_mean', 'Na_1', 'Na_2'
#             , 'Na_mean', 'PaCO2_1', 'PaCO2_2', 'PaCO2_mean'
#             , 'PaO2_1', 'PaO2_2', 'PaO2_mean', 'Platelets_1'
#             , 'Platelets_2', 'Platelets_mean', 'SaO2_1', 'SaO2_2'
#             , 'SaO2_mean', 'SysABP_1', 'SysABP_2', 'SysABP_mean'
#             , 'Temp_1', 'Temp_2', 'Temp_mean', 'Urine_1', 'Urine_2'
#             , 'Urine_mean', 'WBC_1', 'WBC_2', 'WBC_mean', 'SAPS.I', 'SOFA')

## PCA SET A ##
# columns = c("Age", "Height", "ALP_mean", "AST_mean"
# 			, "Albumin_mean", "BUN_mean", "Bilirubin_mean", "Creatinine_mean"
# 			, "DiasABP_mean", "FiO2_mean", "GCS_mean", "Glucose_mean"
# 			, "HCO3_mean", "HCT_mean", "K_mean", "Lactate_mean"
# 			, "MAP_mean", "Mg_mean", "NIDiasABP_mean", "NIMAP_mean"
# 			, "NISysABP_mean", "Na_mean", "PaCO2_mean", "Platelets_mean"
# 			, "SysABP_mean", "Temp_mean", "Urine_mean", "SAPS.I"
# 			, "SOFA", "ALT_mean", "HR_mean", "PaO2_mean"
# 			, "SaO2_mean", "Weight", "WBC_mean")

## FA1 VARIMAX SET A ##
# columns = c("NIDiasABP_mean", "NIMAP_mean", "NISysABP_mean", "DiasABP_mean"
# 			, "MAP_mean", "AST_mean", "ALT_mean", "HCO3_mean"
# 			, "PaCO2_mean", "BUN_mean", "SAPS.I", "SysABP_mean"
# 			, "GCS_mean", "HR_mean", "Urine_mean", "K_mean"
# 			, "Mg_mean")

## FA2 VARIMAX SET A ##
# columns = c("DiasABP_mean", "MAP_mean", "NIDiasABP_mean", "NIMAP_mean"
# 			, "AST_mean", "ALT_mean", "HCO3_mean", "PaCO2_mean"
# 			, "NISysABP_mean", "HR_mean", "K_mean")

## FA2 VARIMAX SET A + Tukey + categoricals ##
# columns = c("DiasABP_mean", "MAP_mean", "NIDiasABP_mean", "NIMAP_mean"
# 			, "AST_mean", "ALT_mean", "HCO3_mean", "PaCO2_mean"
# 			, "NISysABP_mean", "HR_mean", "K_mean", "Albumin_mean"
# 			, "GCS_mean", "Age", "BUN_mean", "Na_mean", "ICUType"
# 			, "Gender", "MechVent")

## FA1 OBLIMIN SET A ##
# columns = c("NIDiasABP_mean", "NISysABP_mean", "SysABP_mean", "MAP_mean"
# 			, "DiasABP_mean", "HCT_mean", "Albumin_mean", "SOFA"
# 			, "Na_mean", "Bilirubin_mean", "HCO3_mean", "Urine_mean")

## FA2 OBLIMIN SET A ##
# columns = c("Na_mean", "HCT_mean", "Urine_mean", "HCO3_mean")

## FA2 OBLIMIN SET A+B ##
#"SysABP_mean", "HCT_mean", "Temp_mean", "Bilirubin_mean"

## FA1 VARIMAX SET A+B+C ##
#"NIDiasABP_mean", "NIMAP_mean", "NISysABP_mean", "DiasABP_mean"
#	  	, "MAP_mean", "AST_mean", "ALT_mean", "BUN_mean"
#		, "Creatinine_mean", "HCO3_mean", "PaCO2_mean", "GCS_mean"
#		, "SysABP_mean", "SAPS.I", "Temp_mean", "Urine_mean"
#		, "FiO2_mean"

## FA2 VARIMAX SET A+B+C ##
#"NIDiasABP_mean", "NIMAP_mean", "NISysABP_mean", "DiasABP_mean"
#	  	, "MAP_mean", "AST_mean", "ALT_mean", "BUN_mean"
#		, "PaCO2_mean", "SysABP_mean", "Albumin_mean"

## FA1 OBLIMIN SET A+B+C ##
#"Albumin_mean", "SysABP_mean", "DiasABP_mean", "Age"
#	  	, "HCT_mean", "K_mean", "BUN_mean", "MAP_mean"
#		, "Urine_mean", "Height", "Bilirubin_mean", "HCO3_mean"

## FA2 OBLIMIN SET A+B+C ##
#"Age", "HCT_mean", "Albumin_mean", "K_mean", "SysABP_mean"

## INDIVIDUAL'S VARIABLES TEST SET A ##
#'Age', 'BUN_mean', 'GCS_mean', 'Temp_mean', 'Urine_mean'

### TESTES MÉTODO FERNANDO ###

## FA2 OBLIMIN SET-A + BUN, GCS ##
# columns = c("Na_mean", "HCT_mean", "Urine_mean", "HCO3_mean", "BUN_mean", "GCS_mean")

## FA2 OBLIMIN SET-A + BUN, GCS + ALP, Albumin, SysABP, AST ##
# columns = c("Na_mean", "HCT_mean", "Urine_mean", "HCO3_mean", "BUN_mean", "GCS_mean"
#             , "ALP_mean"
#             , "Albumin_mean"
#             , "SysABP_mean"
#             , "AST_mean"
#           )

## FA2 OBLIMIN SET-A + BUN, GCS + all ##
columns = c("Na_mean", "HCT_mean", "Urine_mean", "HCO3_mean", "BUN_mean", "GCS_mean"
            , "Age"
            # , "Height"
            # , "Bilirubin_mean"
            , "Creatinine_mean"
            # , "DiasABP_mean"
            # , "FiO2_mean"
            , "Glucose_mean"
            , "K_mean"
            # , "Lactate_mean"
            # , "MAP_mean"
            , "Mg_mean"
            , "NIDiasABP_mean"
            , "NIMAP_mean"
            , "NISysABP_mean"
            , "PaCO2_mean"
            , "Platelets_mean"
            , "Temp_mean"
            , "SAPS.I"
            , "SOFA"
            # , "ALT_mean"
            , "HR_mean"
            # , "PaO2_mean"
            # , "SaO2_mean"
            , "Weight"
            , "WBC_mean"
          )

## CATEGORICAL VARIABLES
columns = append(columns, c("ICUType", "Gender", "MechVent"))

columns = append(columns, "outcome")

result = slice_dataset(set
, columns=columns
  , z_scale = TRUE
# , first_cut_threshold=0.7
  , drop_na_first = FALSE)

write.csv(result, "./tested_datasets/set-c_4sigmas_complete_fa2_oblimin+BUN_GCS_rest(set-a).csv")