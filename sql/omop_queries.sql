select omop.visit_detail.visit_detail_id
	, type.concept_name as visit_type_value
	, detail.concept_name as visit_detail_value
	, omop.care_site.place_of_service_source_value as care_site_value
	, extract(epoch from visit_end_datetime  - visit_start_datetime)/60.0/60.0/24.0 as los
	, extract(epoch from visit_start_datetime  - birth_datetime)/60.0/60.0/24.0/365.0 as age
	, omop.person.gender_source_value
	, case when (death_datetime is not null) then 1 else 0 end as outcome
	, omop.person.person_id
from omop.visit_detail
join omop.concept as type
	on type.concept_id = omop.visit_detail.visit_type_concept_id
join omop.concept as detail
 	on detail.concept_id = omop.visit_detail.visit_detail_concept_id
join omop.care_site using (care_site_id)
join omop.person using (person_id)
left join omop.death using (person_id)
where extract(epoch from visit_end_datetime  - visit_start_datetime)/60.0/60.0/24.0 <= 2
	and type.concept_id = 2000000006
	and detail.concept_id = 581382
	and place_of_service_concept_id in (40481392, 4305366, 4138949, 4149943) -- medical, surgical, cardiac and coronary ICUs
	and extract(epoch from visit_start_datetime  - birth_datetime)/60.0/60.0/24.0/365.0 >= 15
	
select omop.measurement.measurement_concept_id
	, case
		when omop.measurement.measurement_concept_id <> 0
			then snomed.concept_name 
			else dfault.concept_name 
		end as measurement_name
	, case 
		when omop.measurement.measurement_concept_id <> 0
			then snomed.domain_id
			else dfault.domain_id
		end as measurement_meta
	, ty.concept_name as measurement_type
	, o.concept_name as measurement_operator
	, u.concept_name as measurement_unit
	, omop.measurement.unit_source_value
	, valu.concept_name as value_name
	, omop.measurement.value_as_number
	, omop.measurement.measurement_source_value
	, omop.measurement.visit_detail_id
	, omop.measurement.measurement_datetime
from omop.measurement
join omop.concept as snomed
	on omop.measurement.measurement_concept_id = snomed.concept_id
join omop.concept as dfault
	on omop.measurement.measurement_source_concept_id = dfault.concept_id
join omop.concept as ty
	on omop.measurement.measurement_type_concept_id = ty.concept_id
join omop.concept as o
	on omop.measurement.operator_concept_id = o.concept_id
join omop.concept as u
	on omop.measurement.unit_concept_id = u.concept_id
left join omop.concept as valu
	on omop.measurement.value_as_concept_id = valu.concept_id
limit 10;

with conceptualized_measurements as (
	select distinct omop.measurement.measurement_concept_id
		, omop.measurement.measurement_source_concept_id 
	from omop.measurement
), conceptualized_procedure_occurrences as (
	select distinct omop.procedure_occurrence.procedure_source_concept_id as concept_id from omop.procedure_occurrence
), default_measurements as (
	select distinct omop.measurement.measurement_source_concept_id as concept_id 
	from omop.measurement 
	where omop.measurement.measurement_concept_id <> 0
		and omop.measurement.measurement_source_concept_id is not null
), distinct_concept_measurements as (
	select omop.concept.concept_id
		, omop.concept.concept_name
		, omop.concept.domain_id
		, omop.concept.vocabulary_id
	from omop.concept
	join conceptualized_measurements using (concept_id)
	union
	select omop.concept.concept_id
		, omop.concept.concept_name
		, omop.concept.domain_id
		, omop.concept.vocabulary_id
	from omop.concept
	join default_measurements using (concept_id)
), distinct_concept_procedure_occurrences as (
	select omop.concept.concept_id
		, omop.concept.concept_name
		, omop.concept.domain_id
		, omop.concept.vocabulary_id
	from omop.concept
	join conceptualized_procedure_occurrences using (concept_id)
), variables_of_interest as (
	-- invasive mean arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Invasive mean arterial blood pressure' as expected_label
		, 'invasive mean blood pressure, invasive mean arterial pressure, mean blood pressure, mean arterial pressure, MAP' as synonyms
		, '~* [((invasive)?(mean.*(blood|arterial).*pressure)|[^a-zA-Z0-9]+map[^a-zA-Z0-9]+)], !~* [non[^a-zA-Z0-9]?invasive]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (( concept_name ~* 'mean'
			and concept_name ~* '(blood|arterial)'
			and concept_name ~* 'pressure'
			and concept_name ~* '(invasive)?'
		) or ( concept_name ~* 'mean'
			and concept_name ~* '[^a-zA-Z0-9]+a?bp[^a-zA-Z0-9]+'
		)) and concept_name !~* '(non[^a-zA-Z0-9]?invasive)'
	union
	-- invasive diastolic arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Invasive diastolic arterial blood pressure' as expected_label
		, 'invasive diastolic blood pressure, invasive diastolic arterial pressure, diastolic blood pressure, diastolic arterial pressure' as synonyms
		, '~* [(invasive)?.*(diastolic.*(blood|arterial).*pressure)], !~* [non[^a-zA-Z0-9]?invasive]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (( concept_name ~* 'diastol(e|ic)'
			and concept_name ~* '(blood|arterial)'
			and concept_name ~* 'pressure'
			and concept_name ~* '(invasive)?'
		) or ( concept_name ~* 'diastol(e|ic)'
			and concept_name ~* '[^a-zA-Z0-9]+a?bp[^a-zA-Z0-9]+'
		)) and concept_name !~* '(non[^a-zA-Z0-9]?invasive)'
	union
	-- invasive systolic arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Invasive systolic arterial blood pressure' as expected_label
		, 'invasive systolic blood pressure, invasive systolic arterial pressure, systolic blood pressure, systolic arterial pressure' as synonyms
		, '~* [(invasive)?.*(systolic.*(blood|arterial).*pressure)], !~* [non[^a-zA-Z0-9]?invasive]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (( concept_name ~* 'systol(e|ic)'
			and concept_name ~* '(blood|arterial)'
			and concept_name ~* 'pressure'
			and concept_name ~* '(invasive)?'
		) or ( concept_name ~* 'systol(e|ic)'
			and concept_name ~* '[^a-zA-Z0-9]+a?bp[^a-zA-Z0-9]+'
		)) and concept_name !~* '(non[^a-zA-Z0-9]?invasive)'
	union
	-- non-invasive mean arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Non-invasive mean arterial blood pressure' as expected_label
		, 'non-invasive mean blood pressure, non-invasive mean arterial pressure' as synonyms
		, '~* [(mean.*(blood|arterial).*pressure), (non[^a-zA-Z0-9]?invasive)]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'mean'
		and concept_name ~* '(blood|arterial)'
		and concept_name ~* 'pressure'
		and concept_name ~* '(non[^a-zA-Z0-9]?invasive)'
	  ) or ( concept_name ~* 'mean'
		and concept_name ~* 'na?bp')
	union
	-- non-invasive diastolic arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Non-invasive diastolic arterial blood pressure' as expected_label
		, 'non-invasive diastolic blood pressure, non-invasive diastolic arterial pressure' as synonyms
		, '~* [(diastolic.*(blood|arterial).*pressure), (non[^a-zA-Z0-9]?invasive)]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'diastol(e|ic)'
		and concept_name ~* '(blood|arterial)'
		and concept_name ~* 'pressure'
		and concept_name ~* '(non[^a-zA-Z0-9]?invasive)'
	  ) or ( concept_name ~* 'diastol(e|ic)'
		and concept_name ~* 'na?bp')
	union
	-- non-invasive systolic arterial blood pressure
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Non-invasive systolic arterial blood pressure' as expected_label
		, 'non-invasive systolic blood pressure, non-invasive systolic arterial pressure' as synonyms
		, '~* [(systolic.*(blood|arterial).*pressure), (non[^a-zA-Z0-9]?invasive)]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'systol(e|ic)'
		and concept_name ~* '(blood|arterial)'
		and concept_name ~* 'pressure'
		and concept_name ~* '(non[^a-zA-Z0-9]?invasive)'
	  ) or ( concept_name ~* 'systol(e|ic)'
		and concept_name ~* 'na?bp')
	union
	-- albumin
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Albumin' as expected_label
		, 'albumin, serum albumin' as synonyms
		, '~* [albumin, (serum)?]' as used_regex
		, 'g/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'albumin'
		and concept_name ~* '(serum)?'
	union
	-- alp
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'ALP' as expected_label
		, 'alkaline phosphatase, alkaline phosphomonoesterase, phosphomonoesterase, glycerophosphatase, ALP' as synonyms
		, '~* [(alkaline.*(phosphatase|phosphomonoesterase)|phosphomonoesterase|glycerophosphatase|[^a-zA-Z0-9]+alp[^a-zA-Z0-9]+)]' as used_regex
		, 'IU/L' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'alkaline'
			and concept_name ~* '(phosphatase|phosphomonoesterase)'
		  ) or concept_name ~* 'phosphomonoesterase|glycerophosphatase|[^a-zA-Z0-9]+alp[^a-zA-Z0-9]+'
	union
	-- alt
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'ALT' as expected_label
		, 'alanine aminotransferase, alanine transaminase, glutamic pyruvic transaminase, glutamic alanine transaminase, glutamate pyruvate transaminase, glycerophosphatase, SGPT, GPT, ALAT, ALT' as synonyms
		, '~* [(alanine.*(aminotransferase|transaminase)|glutamic.*(pyruvic|alanine).*transaminase|glutamate.*pyruvate.*transaminase|glycerophosphatase|[^a-zA-Z0-9]+(sgpt|gpt|alat|alt)[^a-zA-Z0-9]+)]' as used_regex
		, 'IU/L' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'alanine'
			and concept_name ~* '(aminotransferase|transaminase)'
		  ) or ( concept_name ~* 'glutamic'
			and concept_name ~* '(pyruvic|alanine)'
			and concept_name ~* 'transaminase'
		  ) or ( concept_name ~* 'glutamate'
			and concept_name ~* 'pyruvate'
			and concept_name ~* 'transaminase'
		  ) or concept_name ~* 'glycerophosphatase|[^a-zA-Z0-9]+(sgpt|gpt|alat|alt)[^a-zA-Z0-9]+'
	union
	-- ast "got" foi removido
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'AST' as expected_label
		, 'aspartate aminotransferase, aspartate transaminase, glutamic oxaloacetic transaminase, glutamic aspartic transaminase, glutamate oxaloacetate transaminase, glycerophosphatase, SGOT, ASAT, AST' as synonyms
		, '~* [(aspartate.*(aminotransferase|transaminase)|glutamic.*(oxaloacetic|aspartic).*transaminase|glutamate.*oxaloacetate.*transaminase|glycerophosphatase|[^a-zA-Z0-9]+(sgot|asat|ast)[^a-zA-Z0-9]+)]' as used_regex
		, 'IU/L' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'aspartate'
			and concept_name ~* '(aminotransferase|transaminase)'
		  ) or ( concept_name ~* 'glutamic'
			and concept_name ~* '(oxaloacetic|aspartic)'
			and concept_name ~* 'transaminase'
		  ) or ( concept_name ~* 'glutamate'
			and concept_name ~* 'oxaloacetate'
			and concept_name ~* 'transaminase'
		  ) or concept_name ~* '(glycerophosphatase|[^a-zA-Z0-9]+(sgot|asat|ast)[^a-zA-Z0-9]+)'
	union
	-- bilirubin
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Bilirubin' as expected_label
		, 'split bilirubin, conjugated bilirubin, unconjugated bilirubin, split bilirubin delta, conjugated bilirubin delta, unconjugated bilirubin delta, BU' as synonyms
		, '~* [((split|(un)?conjugated)?.*bilirubin.*(delta)?|[^a-zA-Z0-9]+bu[^a-zA-Z0-9]+)]' as used_regex
		, 'mg/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* '(split|(un)?conjugated)?'
			and concept_name ~* 'bilirubin'
			and concept_name ~* '(delta)?'
		  ) or concept_name ~* '[^a-zA-Z0-9]+bu[^a-zA-Z0-9]+'
	union
	-- bun
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'BUN' as expected_label
		, 'blood urea nitrogen, BUN' as synonyms
		, '~* [(blood.*urea.*nitrogen|[^a-zA-Z0-9]+bun[^a-zA-Z0-9]+)]' as used_regex
		, 'mg/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'blood'
			and concept_name ~* 'urea'
			and concept_name ~* 'nitrogen'
		  ) or concept_name ~* '[^a-zA-Z0-9]+bun[^a-zA-Z0-9]+'
	union
	-- cholesterol
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Cholesterol' as expected_label
		, 'cholesterol' as synonyms
		, '~* [cholesterol]' as used_regex
		, 'mg/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'cholesterol'
	union
	-- creatinine
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Creatinine' as expected_label
		, 'creatinine' as synonyms
		, '~* [creatinine], !~* [ratio]' as used_regex
		, 'mg/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'creatinine'
	union
	-- fio2
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'FiO2' as expected_label
		, 'inspired oxygen, FiO2' as synonyms
		, '~* [(inspired.*oxygen|[^a-zA-Z0-9]+fio2[^a-zA-Z0-9]+)]' as used_regex
		, '[0-1]' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'inspired'
			and concept_name ~* '(oxygen|o2)'
		   ) or concept_name ~* '[^a-zA-Z0-9]+fio2[^a-zA-Z0-9]+'
	union
	-- gcs
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'GCS' as expected_label
		, 'glasgow coma score, CGS' as synonyms
		, '~* [(glasgow.*coma.*score|[^a-zA-Z0-9]+gcs[^a-zA-Z0-9]+)]' as used_regex
		, '[3/15]' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'glasgow'
			and concept_name ~* 'coma'
			and concept_name ~* 'score'
		  ) or concept_name ~* '[^a-zA-Z0-9]+gcs[^a-zA-Z0-9]+'
	union
	-- glucose
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Glucose' as expected_label
		, 'glucose, d-glucose, dextrose' as synonyms
		, '~* [(glasgow.*coma.*score|[^a-zA-Z0-9]+gcs[^a-zA-Z0-9]+)]' as used_regex
		, 'mg/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(d[^a-zA-Z0-9]?)?glucose|dextrose'
	union
	-- hco3
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'HCO3' as expected_label
		, 'bicarbonate, HCO3' as synonyms
		, '~* [(bicarbonate|[^a-zA-Z0-9]+hco3[^a-zA-Z0-9]+)]' as used_regex
		, 'mmol/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(bicarbonate|[^a-zA-Z0-9]+hco3[^a-zA-Z0-9]+)'
	union
	-- hct
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'HCT' as expected_label
		, 'hematocrit, haematocrit, HCT' as synonyms
		, '~* [(ha?ematocrit|[^a-zA-Z0-9]+hct[^a-zA-Z0-9]+)]' as used_regex
		, '%' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(ha?ematocrit|[^a-zA-Z0-9]+hct[^a-zA-Z0-9]+)'
	union
	-- heart rate
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Heart Rate' as expected_label
		, 'heart rate, cardiac rate, HR' as synonyms
		, '~* [((heart|cardiac).*rate|[^a-zA-Z0-9]+hr[^a-zA-Z0-9]+)]' as used_regex
		, 'bpm' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'heart|cardiac'
			and concept_name ~* 'rate'
		  ) or concept_name ~* '[^a-zA-Z0-9]+hr[^a-zA-Z0-9]+'
	union
	-- k
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'K' as expected_label
		, 'potassium, K' as synonyms
		, '~* [(potassium|[^a-zA-Z0-9]+k[^a-zA-Z0-9]+)]' as used_regex
		, 'mEq/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(potassium|[^a-zA-Z0-9]+k[^a-zA-Z0-9]+)'
	union
	-- lactate
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Lactate' as expected_label
		, 'lactate' as synonyms
		, '~* [lactate]' as used_regex
		, 'mmol/dL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'lactate'
	union
	-- mg
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Mg' as expected_label
		, 'magnesium, Mg' as synonyms
		, '~* [(magnesium|[^a-zA-Z0-9]+mg[^a-zA-Z0-9]+)]' as used_regex
		, 'mmol/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(magnesium|[^a-zA-Z0-9]+mg[^a-zA-Z0-9]+)'
	union
	-- mechvent
	select distinct_concept_procedure_occurrences.concept_id
		, distinct_concept_procedure_occurrences.concept_name
		, distinct_concept_procedure_occurrences.domain_id
		, distinct_concept_procedure_occurrences.vocabulary_id
		, 'MechVent' as expected_label
		, '(artificial OR mechanical OR artificially OR mechanically) AND (assist OR assisted OR assistence) AND (ventilation OR respiration OR breathing)' as synonyms
		, '~* [(artificial|mechanical)(ly)?.*(assist)?(ed|ence)?.*(ventilation|respiration|breathing)]' as used_regex
		, '[0/1]' as expected_measurement_unit
	from distinct_concept_procedure_occurrences
	where concept_name ~* '(artificial|mechanical)(ly)?.*(assist)?(ed|ence)?.*(ventilation|respiration|breathing)'
	union
	-- na
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Na' as expected_label
		, 'sodium, Na' as synonyms
		, '~* [(sodium|[^a-zA-Z0-9]+na[^a-zA-Z0-9]+)]' as used_regex
		, 'mEq/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(sodium|[^a-zA-Z0-9]+na[^a-zA-Z0-9]+)'
	union
	-- paco2
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'PaCO2' as expected_label
		, 'arterial* partial* presure carbon dioxide tension*, PaCO2, PCO2' as synonyms
		, '~* [((arterial)?.*(partial)?.*(pressure).*carbon.*dioxide.*(tension)?|[^a-zA-Z0-9]+pa?co2[^a-zA-Z0-9]+)]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* '(arterial)?'
			and concept_name ~* 'partial'
			and concept_name ~* 'pressure'
			and concept_name ~* '(carbon.*dioxide|co2)'
			and concept_name ~* '(tension)?'
		  ) or concept_name ~* '[^a-zA-Z0-9]+pa?co2[^a-zA-Z0-9]+'
	union
	-- pao2
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'PaO2' as expected_label
		, 'arterial* partial* presure oxygen tension*, PaO2, PO2' as synonyms
		, '~* [((arterial)?.*(partial)?.*(pressure).*oxygen.*(tension)?|[^a-zA-Z0-9]+pa?o2[^a-zA-Z0-9]+)]' as used_regex
		, 'mmHg' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* '(arterial)?'
			and concept_name ~* 'partial'
			and concept_name ~* 'pressure'
			and concept_name ~* '(oxygen|o2)'
			and concept_name ~* '(tension)?'
		   ) or concept_name ~* '[^a-zA-Z0-9]+pa?o2[^a-zA-Z0-9]+'
	union
	-- ph
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'pH' as expected_label
		, 'potential hydrogen, pH' as synonyms
		, '~* [(potential.*hydrogen|[^a-zA-Z0-9]+ph[^a-zA-Z0-9]+)]' as used_regex
		, '[0-14]' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'potential'
			and concept_name ~* 'hydrogen'
		  ) or concept_name ~* '[^a-zA-Z0-9]+ph[^a-zA-Z0-9]+'
	union
	-- platelets
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Platelets' as expected_label
		, 'platelets, thrombocyte, PLT' as synonyms
		, '~* [((platelets?|thrombocyte)|[^a-zA-Z0-9]+plt[^a-zA-Z0-9]+)]' as used_regex
		, 'cells/nL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '((platelets?|thrombocyte)|[^a-zA-Z0-9]+plt[^a-zA-Z0-9]+)'
	union
	-- respiration rate
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Respiration Rate' as expected_label
		, '(respiratory OR breathing) AND (rate OR frequency) OR BR OR RR' as synonyms
		, '~* [((respiratory|breathing).*(rate|frequency)|[^a-zA-Z0-9]+(br|rr)[^a-zA-Z0-9]+)]' as used_regex
		, 'bpm' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'respiratory|breathing'
			and concept_name ~* 'rate|frequency'
		   ) or concept_name ~* '[^a-zA-Z0-9]+(br|rr)[^a-zA-Z0-9]+'
	union
	-- sao2
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'SaO2' as expected_label
		, 'arterial oxygen saturation, SaO2' as synonyms
		, '~* [(arterial.*oxygen.*saturation|[^a-zA-Z0-9]+sao2[^a-zA-Z0-9]+)]' as used_regex
		, '%' as expected_measurement_unit
	from distinct_concept_measurements
	where (concept_name ~* 'arterial'
			and concept_name ~* 'oxygen'
			and concept_name ~* 'saturation'
		  ) or concept_name ~* '[^a-zA-Z0-9]+sao2[^a-zA-Z0-9]+'
	union
	-- sao2
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Temperature' as expected_label
		, 'temperature' as synonyms
		, '~* [(arterial.*oxygen.*saturation|[^a-zA-Z0-9]+sao2[^a-zA-Z0-9]+)]' as used_regex
		, 'Celsius' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'temp(erature)?'
	union
	-- troponin i
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Troponin-I' as expected_label
		, 'troponin i, inhibitory troponin subunit' as synonyms
		, '~* [troponin[^a-zA-Z0-9]?i|inhibitory.*troponin.*(subunit)?]' as used_regex
		, 'ug/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'troponin[^a-zA-Z0-9]?i'
		or (concept_name ~* 'inhibitory'
			and concept_name ~* 'troponin'
			and concept_name ~* '(subunit)?'
	   )
	union
	-- troponin t
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Troponin-T' as expected_label
		, 'troponin t, tropomyosin troponin subunit' as synonyms
		, '~* [troponin[^a-zA-Z0-9]?t|tropomyosin.*troponin.*(subunit)?]' as used_regex
		, 'ug/L' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* 'troponin[^a-zA-Z0-9]?t'
		or (concept_name ~* 'tropomyosin'
			and concept_name ~* 'troponin'
			and concept_name ~* '(subunit)?'
	   )
	union
	-- urine
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Urine Output' as expected_label
		, 'quantity* AND (urine OR urinary) AND (output OR volume OR passed)' as synonyms
		, '~* [(quantity)?.*urin(e|ary).*(output|volume|passed)]' as used_regex
		, 'mL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(quantity)?'
		and concept_name ~* 'urin(e|ary)'
		and concept_name ~* '(output|volume|passed)'
	union
	-- wbc
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'White Blood Cell Count' as expected_label
		, '(white blood cell OR WBC) AND count OR WBC OR WCC' as synonyms
		, '~* [((white.*blood.*cell|[^a-zA-Z0-9]+wbc[^a-zA-Z0-9]+).*count?|[^a-zA-Z0-9]+(wbc|wcc)[^a-zA-Z0-9]+)]' as used_regex
		, 'cells/nL' as expected_measurement_unit
	from distinct_concept_measurements
	where concept_name ~* '(white.*blood.*cell|[^a-zA-Z0-9]+wbc[^a-zA-Z0-9]+)'
		and concept_name ~* 'count?'
		or concept_name ~* '[^a-zA-Z0-9]+(wbc|wcc)[^a-zA-Z0-9]+'
	union
	-- weight
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Weight' as expected_label
		, 'weight' as synonyms
		, '~* [weight]' as used_regex
		, 'kg' as expected_measurement_unit
	from distinct_concept_measurements
	where domain_id = 'Measurement' and concept_name ~* 'weight'
	union
	-- height
	select distinct_concept_measurements.concept_id
		, distinct_concept_measurements.concept_name
		, distinct_concept_measurements.domain_id
		, distinct_concept_measurements.vocabulary_id
		, 'Height' as expected_label
		, 'height' as synonyms
		, '~* [height]' as used_regex
		, 'cm' as expected_measurement_unit
	from distinct_concept_measurements
	where domain_id = 'Measurement' and concept_name ~* 'height'
	order by expected_label, vocabulary_id
)
select max(omop.measurement.measurement_id) 
	, omop.measurement.measurement_concept_id
	-- , variables_of_interest.domain_id
	-- , ty.concept_name as measurement_type
	-- , valu.concept_name as value_name
	-- , omop.measurement.measurement_source_value
	-- , omop.measurement.visit_detail_id
	-- , omop.measurement.measurement_datetime
	, variables_of_interest.used_regex
	, variables_of_interest.vocabulary_id
	, variables_of_interest.synonyms
	, variables_of_interest.expected_label
	, variables_of_interest.concept_name
	-- , o.concept_name as measurement_operator
	, u.concept_name as measurement_unit
	, omop.measurement.unit_source_value
	, variables_of_interest.expected_measurement_unit
	, omop.measurement.value_as_number
from omop.measurement
join variables_of_interest
	on omop.measurement.measurement_concept_id = variables_of_interest.concept_id
join omop.concept as ty
	on omop.measurement.measurement_type_concept_id = ty.concept_id
join omop.concept as o
	on omop.measurement.operator_concept_id = o.concept_id
join omop.concept as u
	on omop.measurement.unit_concept_id = u.concept_id
left join omop.concept as valu
	on omop.measurement.value_as_concept_id = valu.concept_id
group by omop.measurement.measurement_concept_id
-- , variables_of_interest.domain_id
-- , ty.concept_name as measurement_type
-- , valu.concept_name as value_name
-- , omop.measurement.measurement_source_value
-- , omop.measurement.visit_detail_id
-- , omop.measurement.measurement_datetime
, variables_of_interest.used_regex
, variables_of_interest.vocabulary_id
, variables_of_interest.synonyms
, variables_of_interest.expected_label
, variables_of_interest.concept_name
-- , o.concept_name as measurement_operator
, u.concept_name
, omop.measurement.unit_source_value
, variables_of_interest.expected_measurement_unit
, omop.measurement.value_as_number;