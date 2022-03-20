with first_admit_2day_adults as (
	select p.subject_id
		, p.gender
		, p.dob
		, min( round( (cast(admittime as date) - cast(dob as date)) / 365.242,2 ) ) as first_admit_age
		, ist.los
		, ist.first_careunit
		, min(a.admittime) as first_admittime
		, case when a.deathtime is not null then 1 else 0 end as outcome
		, a.deathtime
	from mimiciii.patients p
	inner join mimiciii.admissions a
		on p.subject_id = a.subject_id
	inner join mimiciii.icustays as ist
		on p.subject_id = ist.subject_id
	where ist.los <= 2
		and ist.first_careunit = ist.last_careunit
		and ist.first_careunit in ('SICU', 'MICU', 'CSRU', 'CCU')
	GROUP by p.subject_id
		, p.gender
		, p.dob
		, a.deathtime
		, ist.los
		, ist.first_careunit
	having min( round( (cast(admittime as date) - cast(dob as date)) / 365.242,2 ) ) >= 14
	order by p.subject_id
)
select count(*)::float / (select count(*) from first_admit_2day_adults) pct
	, count(*)
	, fa2da.outcome
from first_admit_2day_adults fa2da
group by fa2da.outcome;

with icu_types as (
	select distinct ist.first_careunit  as icutype
	from mimiciii.icustays ist
)
select case
		when icutype = 'MICU' then 'Medical Intensive Care Unit'
		when icutype = 'SICU' then 'Surgical Intensive Care Unit'
		when icutype = 'TSICU' then 'Trauma Surgical Intensive Care Unit'
		when icutype = 'CSRU' then 'Cardiac Surgery Recovery Unit'
		when icutype = 'NICU' then 'Neonatal Intensive Care Unit'
		when icutype = 'CCU' then 'Coronary Care Unit'
	end icu_definition
	, icutype
from icu_types

-- arterial blood pressure
select *
from mimiciii.d_items it
where ( lower(label) like '%pres%' and lower(label) like '%cuff%'
		or lower(label) like '%pres%' and lower(label) like '%bld%'
		or lower(label) like '%pres%' and lower(label) like '%blood%'
		or lower(label) like '%pres%' and lower(label) like '%bladder%'
		or lower(label) like '%pres%' and lower(label) like '%hi%'
		or lower(label) like '%pres%' and lower(label) like '%abd%'
		or lower(label) like '%pres%' and lower(label) like '%cranial%'
		or lower(label) like '%pres%' and lower(label) like '%art%'
		or lower(label) like '%pres%' and lower(label) like '%low%'
		or lower(label) like '%pres%' and lower(label) like '%lumbar%'
		or lower(label) like '%pres%' and lower(label) like '%tank%'
		or lower(label) like '%pres%' and lower(label) like '%mean%'
		or lower(label) like '%pres%' and lower(label) like '%systol%'
		or lower(label) like '%pres%' and lower(label) like '%diastol%'
		or lower(label) like '%pres%' and lower(label) like '%pericardial%'
	    or lower(label) like '%bp%'
		or lower(label) like '%mm%hg%'
	    or lower(label) like '%bp%'
	    or lower(label) like '% map %'
	   ) and lower(label) not like '%bpm%'
   	and lower(label) not like '%effluent%'
   	and lower(label) not like '%plamapheresis%'
	and lower(label) not like '%cbp%'
   	and lower(label) not like '%brbpr%'
   	and lower(label) not like '%potassium%'
   	and lower(label) not like '%sodium%'
   	and lower(label) not like '%iabp%'
   	and lower(label) not like '%aprv%'
	and lower(label) not like '%heliox%'
	and lower(label) not like '%he%'
	and lower(label) not like '%airway%'
   	or lower(label) like '%iabp%mean%'
order by label

-- albumin
select *
from mimiciii.d_items it
where lower(label) like '%albumin%'
order by label

-- alp
select *
from mimiciii.d_items it
where lower(label) like '%alk%'
	and lower(label) not like '%walking%'
order by label

-- alt
select *
from mimiciii.d_items it
where lower(label) like 'alt'
order by label

-- ast
select *
from mimiciii.d_items it
where lower(label) like 'ast'
order by label

-- bilirubin
select *
from mimiciii.d_items it
where lower(label) like '%bili%'
	and lower(label) not like '%mobili%'
	and lower(label) not like '%rehab%'
	and lower(label) not like '%mirab%'
	and lower(label) not like '%umbil%'
order by label

-- bun
select *
from mimiciii.d_items it
where lower(label) like '%bun%'
order by label

-- cholesterol
select *
from mimiciii.d_items it
where lower(label) like '%cholesterol%'
order by label

-- creatinine
select *
from mimiciii.d_items it
where lower(label) like '%creatinine%'
order by label

-- fio2
select *
from mimiciii.d_items it
where lower(label) like '%fio2%'
	or lower(label) like '%fi02%'
	or lower(label) like '%inspired o2 fraction%'
order by label

-- gcs
select *
from mimiciii.d_items it
where lower(label) like '%gcs%'
order by label

-- glucose
select *
from mimiciii.d_items it
where (lower(label) like '%glucose%'
	or lower(label) like '%glu%')
	and lower(label) not like '%ca%'
	and lower(label) not like '%glubionate%'
	and lower(label) not like '%gluconate%'
	and lower(label) not like '%glucagon%'
	and lower(label) not like '%glucerna%'
	and lower(label) not like '%glulateral%'
order by label

-- hco3
select *
from mimiciii.d_items it
where (lower(label) like '%hco3%'
	or lower(label) like '%bicarbonate%'
	or lower(label) like '%bi%car%')
	and lower(label) not like '%na%'
	and lower(label) not like '%sod%'
	and lower(label) not like '%umbi%'
order by label

-- hct
select *
from mimiciii.d_items it
where lower(label) like '%hct%'
	or lower(label) like '%hematocrit%'
order by label

-- hr
select *
from mimiciii.d_items it
where (lower(label) like '% hr %'
	or lower(label) like '%heart%rate%')
	and lower(label) not like '%mg%'
order by label

-- k serum
select *
from mimiciii.d_items it
where (lower(label) like '% k %'
	or lower(label) like '%meqk%'
	or lower(label) like '%potassium%'
	or lower(label) like '%potass%')
	and lower(label) not like '%cl%'
	and lower(label) not like '%phos%'
	and lower(label) not like '%penicillin%'
	and lower(label) not like '%acetate%'
	and lower(label) not like '%chl%'
order by label

-- lactate
select *
from mimiciii.d_items it
where lower(label) like '%lactate%'
	or lower(label) like '%lactic%'
order by label

-- mg serum
select *
from mimiciii.d_items it
where lower(label) like '%magnesium%'
order by label

-- mechenical ventilation
select *
from mimiciii.d_items it
where (lower(label) like '%ven%'
	or lower(label) like '%mechanically ventilated%'
	or lower(label) like '%mv%'
	or lower(label) like '%ventilation%'
	or lower(label) like '% ve %')
	and lower(label) not like '%atr%'
	and lower(label) not like '%blood%'
	and lower(label) not like '%ig%'
	and lower(label) not like '%immunology%'
	and lower(label) not like '%left%'
	and lower(label) not like '%right%'
	and lower(label) not like '%ventricular%'
	and lower(label) not like '%vo2%'
	and lower(label) not like '%urine%'
	and lower(label) not like '%venogram%'
	and lower(label) not like '%venous%'
	and lower(label) not like '%ven0us%'
	and lower(label) not like '%simv%'
order by label

-- PAREI AQUI
-- na serum
select *
from mimiciii.d_items it
where (lower(label) like '% na %'
	or lower(label) like '%sodium%'
	or lower(label) like '%sod%')
	and lower(label) not like '%phos%'
	and lower(label) not like '%ace%'
	and lower(label) not like '%bicarb%'
	and lower(label) not like '%hco3%'
order by label

-- paco2
select *
from mimiciii.d_items it
where lower(label) like '%pco2%'
order by label

-- pao2
select *
from mimiciii.d_items it
where lower(label) like '%po2%'
	or lower(label) like '%pao2%'
	or lower(label) like '%pa%o2%'
order by label

-- ph
select *
from mimiciii.d_items it
where lower(label) like '%ph%'
order by label

--platelets
select *
from mimiciii.d_items it
where lower(label) like '%platelet%'
order by label

-- respiration rate
select *
from mimiciii.d_items it
where lower(label) like '%resp%'
	or lower(label) like '%rr%'
order by label

-- sao2
select *
from mimiciii.d_items it
where lower(label) like '%o2 sat%'
	or lower(label) like '%o2%sat%'
	or lower(label) like '%sao2%'
order by label

-- temperature
select *
from mimiciii.d_items it
where lower(label) like '%temp%'
order by label

-- troponin-i and troponin-t
select *
from mimiciii.d_items it
where lower(label) like '%troponin%'
order by label

-- urine output
select *
from mimiciii.d_items it
where lower(label) like '%urine%'
order by label

-- wbc
select *
from mimiciii.d_items it
where lower(label) like '%wbc%'
	or lower(label) like '%whiteblood%'
order by label

-- weight
select *
from mimiciii.d_items it
where lower(label) like '%weight%'
	or lower(label) like '%wt%'
order by label

-- apache
select *
from mimiciii.d_items it
where lower(label) like '%apache%'
order by label

-- height
select *
from mimiciii.d_items it
where lower(label) like '%height%'
	or lower(label) like '%length%'
order by label

-- sofa score
select *
from mimiciii.d_items it
where lower(label) like '%sofa%'
order by label

-- select *
-- from mimiciii.d_items it
--inner join mimiciii.chartevents ce
--	on it.itemid = ce.itemid
--inner join mimiciii.inputevents_mv mv
	--on it.itemid = mv.itemid
--where upper(it.label) like upper('admission%weight%')
--	or upper(it.label) like upper('admission%wt%')
--	or upper(it.label) like upper('admission%wgt%')
--	or upper(it.label) like upper('admit%weight%')
--	or upper(label) like upper('admit%wt%') 
--	or upper(label) like upper('admit%wgt%')