
-- Metadata specification for STHAM, or the Spatio-Temporal Human Activity Model, OrientDB style

connect remote:workbench.ccts.utah.edu/gmdr admin

--HOUSEHOLD COMPONENTS

--Namespaces
create vertex Namespace set name='STHAM',namespace='stham-alt',description='Spatio-Temporal Human Activity Model';
create vertex Namespace set name='USCensus',namespace='stham-alt',description='United States Census Bureau';
create vertex Namespace set name='USPostalService',namespace='stham-alt',description='United States Postal Service';
create vertex Namespace set name='BLS',namespace='stham-alt',description='Bureau of Labor Statistics';


-- Datasets
create vertex DataSet set name='indvs',namespace='stham-alt',description='Descriptive attributes for individual agents';
create vertex DataSet set name='blockaddr',namespace='stham-alt',description='Addresses assigned to census blocks';
create vertex DataSet set name='CensusTables',namespace='stham-alt',description='Tables from the US Census';
create vertex DataSet set name='UTAddressPoints',namespace='stham-alt',description='US postal addresses for Utah'
create vertex DataSet set name='CensusBlocks',namespace='stham-alt',description='US census block shapefile';
create vertex DataSet set name='ACSTables',namespace='stham-alt',description='American Community Survey Tables';
create vertex DataSet set name='school',namespace='stham-alt',description='School classification for agents';
create vertex DataSet set name='LEHD',namespace='stham-alt',description='Longitudinal Household Employee Dynamic tables';
create vertex DataSet set name='employ',namespace='stham-alt',description='Employment assignment for agents';
create vertex DataSet set name='ATUS',namespace='stham-alt',description='American Time Use Survey';
create vertex DataSet set name='actlabels',namespace='stham-alt',description='Demograhpic and activity profile labels';
create vertex DataSet set name='Demotree',namespace='stham-alt',description='Decision tree for sorting demographic classes';
create vertex Dataset set name='actplots',namespace='stham-alt',description='Activity profile plots';
create vertex DataSet set name='indvlabels',namespace='stham-alt',description='Demographic class labels for agents';
create vertex DataSet set name='weibull',namespace='stham-alt',description='Weibull parameters for activity lengths';
create vertex DataSet set name='actdata',namespace='stham-alt',description='reparsed activity data for modelling activity start times,lengths, and frequencies';
create vertex DataSet set name='labeltabD',namespace='stham-alt',description='transformed special form of labels';
create vertex DataSet set name='Finfluence',namespace='stham-alt',description='Population distribution matrix output';


-- Functional Components

create vertex Function set name='fHouseholdScript',namespace='stham-alt',executable='python';
create vertex Function set name='fSchoolsScript',namespace='stham-alt',executable='python';
create vertex Function set name='fEmploymentScript',namespace='stham-alt',executable='python';
create vertex Function set name='fActprofileScript',namespace='stham-alt',executable='python';
create vertex Function set name='fAggregateClassifyScript',namespace='stham-alt',executable='python';
create vertex Function set name='fLabeltabScript',namespace='stham-alt',executable='python';
create vertex Function set name='fWeibullActScript',namespace='stham-alt',executable='python';
create vertex Function set name='fActivityinstanceScript',namespace='stham-alt',executable='python';
create vertex Function set name='fActmodelScript',namespace='stham-alt',executable='python';

create vertex Transform set name='actmodel',namespace='stham-alt';



--Relationships
create edge INPUT_OF from (select from DataSet where namespace='stham-alt' and name in ['ATUS','ACSTables','LEHD','CensusBlocks','CensusTables','UTAddressPoints']) to (select from Transform where namespace='stham-alt' and name='actmodel');

create edge OUTPUT_OF from (select from Transform where namespace='stham-alt' and name='actmodel') to (select from DataSet where namespace='stham-alt' and name in ['Finfluence','indvs','employ','school','indvlabels','weibull','actdata','labeltabD','actlabels','actplots','Demotree','blockaddr']);

create edge APPLIES from (select from Transform where namespace='stham-alt' and name='actmodel') to (select from Function where namespace='stham-alt' and name in ['fActmodelScript','fHouseholdScript','fSchoolsScript','fEmploymentScript','fActprofileScript','fAggregateClassifyScript','fLabeltabScript','fWeibullActScript','fActivityinstanceScript']);


--Namespace relationships

create edge BELONGS from (select from DataSet where namespace='stham-alt' and name in ['Finfluence','indvs','employ','school','indvlabels','weibull','actdata','labeltabD','actlabels','actplots','Demotree','blockaddr']) to (select from Namespace where namespace='stham-alt' and name='STHAM');

create edge BELONGS from (select from DataSet where namespace='stham-alt' and name in ['UTAddressPoints']) to (select from Namespace where namespace='stham-alt' and name='USPostalService');

create edge BELONGS from (select from DataSet where namespace='stham-alt' and name in ['CensusTables','CensusBlocks','LEHD','ACSTables']) to (select from Namespace where namespace='stham-alt' and name='USCensus');

create edge BELONGS from (select from DataSet where namespace='stham-alt' and name in ['ATUS']) to (select from Namespace where namespace='stham-alt' and name='BLS');