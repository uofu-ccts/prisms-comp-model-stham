
-- Metadata specification for STHAM, or the Spatio-Temporal Human Activity Model, OrientDB style


create vertex Namespace set name='stham',namespace='stham',description='Spatio-Temporal Human Activity Model';


create vertex DataSet set name='indvs',namespace='stham',description='Descriptive attributes for individual agents';
create vertex DataSet set name='blockaddr',namespace='stham',description='Addresses assigned to census blocks';
create vertex DataSet set name='CensusTables',namespace='stham',description='Tables from the US Census';
create vertex DataSet set name='UTAddressPoints',namespace='stham',description='US postal addresses for Utah'
create vertex DataSet set name='CensusBlocks',namespace='stham',description='