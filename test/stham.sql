
-- Metadata specification for STHAM, or the Spatio-Temporal Human Activity Model, OrientDB style

connect remote:workbench.ccts.utah.edu/gmdr admin

--HOUSEHOLD COMPONENTS

--Namespaces
create vertex Namespace set name='STHAM',namespace='stham',description='Spatio-Temporal Human Activity Model';
create vertex Namespace set name='USCensus',namespace='stham',description='United States Census Bureau';
create vertex Namespace set name='USPostalService',namespace='stham',description='United States Postal Service';


-- Datasets
create vertex DataSet set name='indvs',namespace='stham',description='Descriptive attributes for individual agents';
create vertex DataSet set name='blockaddr',namespace='stham',description='Addresses assigned to census blocks';
create vertex DataSet set name='CensusTables',namespace='stham',description='Tables from the US Census';
create vertex DataSet set name='UTAddressPoints',namespace='stham',description='US postal addresses for Utah'
create vertex DataSet set name='CensusBlocks',namespace='stham',description='US census block shapefile';

-- Functional Components
create vertex Transform set name='households',namespace='stham';

create vertex Function set name='fHouseholdScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['CensusTables','UTAddressPoints','CensusBlocks']) to (select from Transform where namespace='stham' and name='households');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='households') to (select from DataSet where namespace='stham' and name in ['indvs','blockaddr']);

create edge APPLIES from (select from Transform where namespace='stham' and name='households') to (select from Function where namespace='stham' and name in ['fHouseholdScript']);


--Namespace relationships
create edge BELONGS from (select from DataSet where namespace='stham' and name in ['CensusTables','CensusBlocks']) to (select from Namespace where namespace='stham' and name='USCensus');

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['UTAddressPoints']) to (select from Namespace where namespace='stham' and name='USPostalService');

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['indvs','blockaddr']) to (select from Namespace where namespace='stham' and name='STHAM');


--SCHOOLS

--Namespaces

--Datasets
create vertex DataSet set name='ACSTables',namespace='stham',description='American Community Survey Tables';
create vertex DataSet set name='school',namespace='stham',description='School classification for agents';

-- Functional Components
create vertex Transform set name='schools',namespace='stham';

create vertex Function set name='fSchoolsScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['ACSTables','indvs']) to (select from Transform where namespace='stham' and name='schools');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='schools') to (select from DataSet where namespace='stham' and name in ['school']);

create edge APPLIES from (select from Transform where namespace='stham' and name='schools') to (select from Function where namespace='stham' and name in ['fSchoolsScript']);


--Namespace relationships
create edge BELONGS from (select from DataSet where namespace='stham' and name in ['ACSTables']) to (select from Namespace where namespace='stham' and name='USCensus');

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['school']) to (select from Namespace where namespace='stham' and name='STHAM');

--EMPLOYMENT

--Namespaces

--Datasets
create vertex DataSet set name='LEHD',namespace='stham',description='Longitudinal Household Employee Dynamic tables';
create vertex DataSet set name='employ',namespace='stham',description='Employment assignment for agents';

-- Functional Components
create vertex Transform set name='employment',namespace='stham';

create vertex Function set name='fEmploymentScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['ACSTables','indvs','LEHD','blockaddr']) to (select from Transform where namespace='stham' and name='employment');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='employment') to (select from DataSet where namespace='stham' and name in ['employ']);

create edge APPLIES from (select from Transform where namespace='stham' and name='employment') to (select from Function where namespace='stham' and name in ['fEmploymentScript']);


--Namespace relationships
create edge BELONGS from (select from DataSet where namespace='stham' and name in ['LEHD']) to (select from Namespace where namespace='stham' and name='USCensus');

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['employ']) to (select from Namespace where namespace='stham' and name='STHAM');

--ACTPROFILE

--Namespaces
create vertex Namespace set name='BLS',namespace='stham',description='Bureau of Labor Statistics';

--Datasets
create vertex DataSet set name='ATUS',namespace='stham',description='American Time Use Survey';
create vertex DataSet set name='actlabels',namespace='stham',description='Demograhpic and activity profile labels';
create vertex DataSet set name='Demotree',namespace='stham',description='Decision tree for sorting demographic classes';
create vertex Dataset set name='actplots',namespace='stham',description='Activity profile plots';

-- Functional Components
create vertex Transform set name='actprofile',namespace='stham';

create vertex Function set name='fActprofileScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['ATUS']) to (select from Transform where namespace='stham' and name='actprofile');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='actprofile') to (select from DataSet where namespace='stham' and name in ['actlabels','actplots','Demotree']);

create edge APPLIES from (select from Transform where namespace='stham' and name='actprofile') to (select from Function where namespace='stham' and name in ['fActprofileScript']);

--Namespace relationships
create edge BELONGS from (select from DataSet where namespace='stham' and name in ['ATUS']) to (select from Namespace where namespace='stham' and name='BLS');

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['actlabels','Demotree','actplots']) to (select from Namespace where namespace='stham' and name='STHAM');

--AGGREGATE CLASSIFY

--Namespaces

--Datasets
create vertex DataSet set name='indvlabels',namespace='stham',description='Demographic class labels for agents';

-- Functional Components
create vertex Transform set name='aggregateclassify',namespace='stham';

create vertex Function set name='fAggregateClassifyScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['indvs','employ','school','Demotree']) to (select from Transform where namespace='stham' and name='aggregateclassify');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='aggregateclassify') to (select from DataSet where namespace='stham' and name in ['indvlabels']);

create edge APPLIES from (select from Transform where namespace='stham' and name='aggregateclassify') to (select from Function where namespace='stham' and name in ['fAggregateClassifyScript']);

--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['indvlabels']) to (select from Namespace where namespace='stham' and name='STHAM');


--WEIBULL

--Namespaces

--Datasets
create vertex DataSet set name='weibull',namespace='stham',description='Weibull parameters for activity lengths';

-- Functional Components
create vertex Transform set name='weibullact',namespace='stham';

create vertex Function set name='fWeibullActScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['ATUS']) to (select from Transform where namespace='stham' and name='weibullact');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='weibullact') to (select from DataSet where namespace='stham' and name in ['weibull']);

create edge APPLIES from (select from Transform where namespace='stham' and name='weibullact') to (select from Function where namespace='stham' and name in ['fWeibullActScript']);

--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['weibull']) to (select from Namespace where namespace='stham' and name='STHAM');


--ACTIVITYINSTANCE

--Namespaces

--Datasets
create vertex DataSet set name='actdata',namespace='stham',description='';

-- Functional Components
create vertex Transform set name='activityinstance',namespace='stham';

create vertex Function set name='fActivityinstanceScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['actlabels','ATUS']) to (select from Transform where namespace='stham' and name='activityinstance');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='activityinstance') to (select from DataSet where namespace='stham' and name in ['actdata']);

create edge APPLIES from (select from Transform where namespace='stham' and name='activityinstance') to (select from Function where namespace='stham' and name in ['fActivityinstanceScript']);

--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['actdata']) to (select from Namespace where namespace='stham' and name='STHAM');


--LABLETAB

--Namespaces

--Datasets
create vertex DataSet set name='labeltabD',namespace='stham',description='';

-- Functional Components
create vertex Transform set name='labeltab',namespace='stham';

create vertex Function set name='fLabeltabScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['actlabels']) to (select from Transform where namespace='stham' and name='labeltab');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='labeltab') to (select from DataSet where namespace='stham' and name in ['labeltabD']);

create edge APPLIES from (select from Transform where namespace='stham' and name='labeltab') to (select from Function where namespace='stham' and name in ['fLabeltabScript']);

--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['labeltabD']) to (select from Namespace where namespace='stham' and name='STHAM');


--ACTMODEL

--Namespaces

--Datasets
create vertex DataSet set name='Finfluence',namespace='stham',description='Population distribution matrix output';

-- Functional Components
create vertex Transform set name='actmodel',namespace='stham';

create vertex Function set name='fActmodelScript',namespace='stham',executable='python';

--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham' and name in ['indvs','employ','school','indvlabels','weibull','actdata','labeltabD']) to (select from Transform where namespace='stham' and name='actmodel');

create edge OUTPUT_OF from (select from Transform where namespace='stham' and name='actmodel') to (select from DataSet where namespace='stham' and name in ['Finfluence']);

create edge APPLIES from (select from Transform where namespace='stham' and name='actmodel') to (select from Function where namespace='stham' and name in ['fActmodelScript']);

--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham' and name in ['Finfluence']) to (select from Namespace where namespace='stham' and name='STHAM');
