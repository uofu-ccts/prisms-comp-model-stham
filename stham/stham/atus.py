import pandas as pd;
import numpy as np;
from datetime import datetime;
import activities
import trajectories
import joblib

def tuccconv(x):
	if(x == '-1'): return -1;
	if(x == '-2'): return -2;
	if(x == '-3'): return -3;
	b = datetime.strptime(x,'%H:%M:%S')
	return b.hour*60 + b.minute;

def loadATUS(datapath,daypick=[1,2,3,4,5,6,7]):

	#ONLY GUARANTEED TO WORK ON 2015 ATUS

	#tiercode = 'TRTIER2'
	tiercode = 'TRCODE'

	#FIXME get rid of automatic paths
	acttable = pd.read_csv(datapath + "atusact_2015/atusact_2015.dat")
	demotable = pd.read_csv(datapath + "atusresp_2015/atusresp_2015.dat")
	rosttable = pd.read_csv(datapath + "atusrost_2015/atusrost_2015.dat")

	rosttable = rosttable[rosttable['TERRP'].apply(lambda x: x in [18,19])]
	rosttable = rosttable.drop(['TXAGE','TXRRP','TXSEX','TULINENO','TERRP'],axis=1)

	demotable = pd.merge(demotable,rosttable,on='TUCASEID')
	acttable = pd.merge(acttable,demotable[['TUCASEID','TUDIARYDAY','TESEX','TEAGE']],on='TUCASEID')

	# What is this? TUCC has string datetimes embedded, need to translate
	demotable['TUCC2'] = demotable['TUCC2'].apply(tuccconv);
	demotable['TUCC4'] = demotable['TUCC4'].apply(tuccconv);

	actmapping = np.sort(list(set(acttable[tiercode])))
	actcount = len(actmapping)
	ati = { tr:i for i,tr in enumerate(actmapping) }

	locmapping = np.sort(list(set(acttable['TEWHERE'])))
	wti = { tr:i for i,tr in enumerate(locmapping) }

	acttable['actcode'] = acttable[tiercode].apply(lambda x: ati[x]);
	acttable = acttable[acttable['TUDIARYDAY'].apply(lambda x: x in daypick) ];
	demotable = demotable[demotable['TUDIARYDAY'].apply(lambda x: x in daypick) ];

	acttable['case'] = acttable['TUCASEID']
	acttable['start'] = (acttable['TUCUMDUR24']-acttable['TUACTDUR24'])
	acttable['end'] = acttable['TUCUMDUR24']
	acttable['length'] = acttable['end']-acttable['start']
	acttable['length'] = acttable['length'].apply(lambda x:1 if x <= 0 else x)
	acttable['where'] = acttable['TEWHERE'].apply(lambda x: wti[x]);


	democols = ['TUCASEID','TEAGE', 'TEHRUSL1', 'TELFS', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESEX', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRHHCHILD', 'TRSPPRES', 'TUDIS2', 'TUELNUM', 'TUSPUSFT']

	return acttable[['case','start','end','length','actcode','where']], demotable[democols], actmapping, locmapping

def test():

	#FIXME: remove hard coded path and code
	datapath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/timeuse/"

	acttable,demotable,actmapping,locmapping = loadATUS(datapath)

	labelpath = "/uufs/chpc.utah.edu/common/home/u0403692/prog/prism/data/final-label-classifier/"
	labels = pd.read_csv(labelpath + "labels.csv")
	caselist = labels[labels["daytypelabelreduce"].values==24]['TUCASEID'].values
	subact = acttable[acttable['case'].apply(lambda x: True if x in caselist else False)]
	window = activities.buildWindow(subact);

	#FIXME: remove dump or add test path
	joblib.dump(window,"windowtest.gz",compress=3)
	print(window)

	# print(trajectories.buildtraj(0,0,window))

	# supervec,supercolumns = activities.vectorizeActs(acttable,demotable,actmapping,,'TUCASEID')



if __name__ == "__main__": 
	test();

	
	##############################################################################
	#supercolumns = ['TUCASEID', 'TULINENO', 'TUYEAR', 'TUMONTH', 'TEABSRSN', 'TEERN', 'TEERNH1O', 'TEERNH2', 'TEERNHRO', 'TEERNHRY', 'TEERNPER', 'TEERNRT', 'TEERNUOT', 'TEERNWKP', 'TEHRFTPT', 'TEHRUSL1', 'TEHRUSL2', 'TEHRUSLT', 'TEIO1COW', 'TEIO1ICD', 'TEIO1OCD', 'TELAYAVL', 'TELAYLK', 'TELFS', 'TELKAVL', 'TELKM1', 'TEMJOT', 'TERET1', 'TESCHENR', 'TESCHFT', 'TESCHLVL', 'TESPEMPNOT', 'TESPUHRS', 'TRCHILDNUM', 'TRDPFTPT', 'TRDTIND1', 'TRDTOCC1', 'TRERNHLY', 'TRERNUPD', 'TRERNWA', 'TRHERNAL', 'TRHHCHILD', 'TRHOLIDAY', 'TRIMIND1', 'TRMJIND1', 'TRMJOCC1', 'TRMJOCGR', 'TRNHHCHILD', 'TRNUMHOU', 'TROHHCHILD', 'TRSPFTPT', 'TRSPPRES', 'TRTALONE', 'TRTALONE_WK', 'TRTCC', 'TRTCCC', 'TRTCCC_WK', 'TRTCCTOT', 'TRTCHILD', 'TRTCOC', 'TRTEC', 'TRTFAMILY', 'TRTFRIEND', 'TRTHH', 'TRTHHFAMILY', 'TRTNOCHILD', 'TRTNOHH', 'TRTO', 'TRTOHH', 'TRTOHHCHILD', 'TRTONHH', 'TRTONHHCHILD', 'TRTSPONLY', 'TRTSPOUSE', 'TRTUNMPART', 'TREMODR', 'TRWERNAL', 'TRYHHCHILD', 'TTHR', 'TTOT', 'TTWK', 'TUABSOT', 'TUBUS', 'TUBUS1', 'TUBUS2OT', 'TUBUSL1', 'TUBUSL2', 'TUBUSL3', 'TUBUSL4', 'TUCC2', 'TUCC4', 'TUCC5B_CK', 'TUCC5_CK', 'TUCC9', 'TUDIARYDATE', 'TUDIARYDAY', 'TUDIS', 'TUDIS1', 'TUDIS2', 'TUECYTD', 'TUELDER', 'TUELFREQ', 'TUELNUM', 'TUERN2', 'TUERNH1C', 'TUFINLWGT', 'TUFWK', 'TUIO1MFG', 'TUIODP1', 'TUIODP2', 'TUIODP3', 'TULAY', 'TULAY6M', 'TULAYAVR', 'TULAYDT', 'TULK', 'TULKAVR', 'TULKDK1', 'TULKDK2', 'TULKDK3', 'TULKDK4', 'TULKDK5', 'TULKDK6', 'TULKM2', 'TULKM3', 'TULKM4', 'TULKM5', 'TULKM6', 'TULKPS1', 'TULKPS2', 'TULKPS3', 'TULKPS4', 'TULKPS5', 'TULKPS6', 'TURETOT', 'TUSPABS', 'TUSPUSFT', 'TUSPWK', 'TXABSRSN', 'TXERN', 'TXERNH1O', 'TXERNH2', 'TXERNHRO', 'TXERNHRY', 'TXERNPER', 'TXERNRT', 'TXERNUOT', 'TXERNWKP', 'TXHRFTPT', 'TXHRUSL1', 'TXHRUSL2', 'TXHRUSLT', 'TXIO1COW', 'TXIO1ICD', 'TXIO1OCD', 'TXLAYAVL', 'TXLAYLK', 'TXLFS', 'TXLKAVL', 'TXLKM1', 'TXMJOT', 'TXRET1', 'TXSCHENR', 'TXSCHFT', 'TXSCHLVL', 'TXSPEMPNOT', 'TXSPUHRS', 'TXTCC', 'TXTCCTOT', 'TXTCOC', 'TXTHH', 'TXTNOHH', 'TXTO', 'TXTOHH', 'TXTONHH', 'TEAGE', 'TESEX', '10101', '10102', '10199', '10201', '10299', '10301', '10399', '10401', '20101', '20102', '20103', '20104', '20199', '20201', '20202', '20203', '20299', '20301', '20302', '20303', '20399', '20401', '20402', '20499', '20501', '20502', '20601', '20602', '20699', '20701', '20799', '20801', '20899', '20901', '20902', '20903', '20904', '20905', '20999', '29999', '30101', '30102', '30103', '30104', '30105', '30106', '30108', '30109', '30110', '30111', '30112', '30199', '30201', '30202', '30203', '30204', '30299', '30301', '30302', '30303', '30399', '30401', '30402', '30403', '30404', '30405', '30499', '30501', '30502', '30503', '30504', '30599', '40101', '40102', '40103', '40104', '40105', '40106', '40108', '40109', '40110', '40111', '40112', '40199', '40201', '40202', '40301', '40302', '40303', '40399', '40401', '40402', '40403', '40404', '40405', '40499', '40501', '40502', '40503', '40504', '40505', '40506', '40507', '40508', '40599', '49999', '50101', '50102', '50103', '50104', '50199', '50201', '50202', '50203', '50299', '50301', '50302', '50303', '50304', '50399', '50401', '50403', '50404', '50499', '59999', '60101', '60102', '60103', '60104', '60199', '60201', '60202', '60203', '60204', '60299', '60301', '60302', '60399', '60401', '60402', '60499', '69999', '70101', '70102', '70103', '70104', '70105', '70201', '80101', '80102', '80201', '80202', '80203', '80301', '80401', '80402', '80403', '80499', '80501', '80502', '80601', '80701', '80702', '89999', '90101', '90102', '90103', '90104', '90199', '90201', '90202', '90299', '90301', '90302', '90399', '90401', '90501', '90502', '90599', '99999', '100101', '100102', '100103', '100199', '100201', '100299', '100304', '100305', '100401', '110101', '110201', '120101', '120201', '120202', '120299', '120301', '120302', '120303', '120304', '120305', '120306', '120307', '120308', '120309', '120310', '120311', '120312', '120313', '120399', '120401', '120402', '120403', '120404', '120405', '120499', '120501', '120502', '120503', '120504', '130101', '130102', '130103', '130104', '130105', '130106', '130107', '130108', '130109', '130110', '130112', '130113', '130114', '130115', '130116', '130117', '130118', '130119', '130120', '130122', '130124', '130125', '130126', '130127', '130128', '130129', '130130', '130131', '130132', '130133', '130134', '130135', '130136', '130199', '130202', '130203', '130207', '130210', '130212', '130213', '130216', '130218', '130219', '130220', '130222', '130223', '130224', '130225', '130226', '130227', '130229', '130232', '130299', '130301', '130302', '139999', '140101', '140102', '140103', '140105', '149999', '150101', '150102', '150103', '150104', '150105', '150106', '150199', '150201', '150202', '150203', '150204', '150299', '150301', '150302', '150399', '150401', '150402', '150501', '150601', '150602', '150699', '150701', '150801', '159999', '160101', '160102', '160103', '160104', '160105', '160106', '160107', '160108', '160199', '160201', '180101', '180201', '180202', '180203', '180204', '180205', '180206', '180207', '180208', '180209', '180299', '180301', '180302', '180303', '180304', '180305', '180401', '180402', '180403', '180404', '180405', '180499', '180501', '180502', '180503', '180504', '180599', '180601', '180602', '180603', '180604', '180699', '180701', '180702', '180703', '180704', '180801', '180802', '180803', '180804', '180805', '180806', '180807', '180899', '180901', '180902', '180903', '180904', '180905', '180999', '181001', '181002', '181101', '181201', '181202', '181203', '181204', '181205', '181299', '181301', '181302', '181399', '181401', '181501', '181599', '181601', '181801', '189999', '500101', '500103', '500105', '500106', '500107']



	# fullcols = goodcols + ['TUDIARYDAY',
	#actcolumns = ['10101', '10102', '10199', '10201', '10299', '10301', '10399', '10401', '20101', '20102', '20103', '20104', '20199', '20201', '20202', '20203', '20299', '20301', '20302', '20303', '20399', '20401', '20402', '20499', '20501', '20502', '20601', '20602', '20699', '20701', '20799', '20801', '20899', '20901', '20902', '20903', '20904', '20905', '20999', '29999', '30101', '30102', '30103', '30104', '30105', '30106', '30108', '30109', '30110', '30111', '30112', '30199', '30201', '30202', '30203', '30204', '30299', '30301', '30302', '30303', '30399', '30401', '30402', '30403', '30404', '30405', '30499', '30501', '30502', '30503', '30504', '30599', '40101', '40102', '40103', '40104', '40105', '40106', '40108', '40109', '40110', '40111', '40112', '40199', '40201', '40202', '40301', '40302', '40303', '40399', '40401', '40402', '40403', '40404', '40405', '40499', '40501', '40502', '40503', '40504', '40505', '40506', '40507', '40508', '40599', '49999', '50101', '50102', '50103', '50104', '50199', '50201', '50202', '50203', '50299', '50301', '50302', '50303', '50304', '50399', '50401', '50403', '50404', '50499', '59999', '60101', '60102', '60103', '60104', '60199', '60201', '60202', '60203', '60204', '60299', '60301', '60302', '60399', '60401', '60402', '60499', '69999', '70101', '70102', '70103', '70104', '70105', '70201', '80101', '80102', '80201', '80202', '80203', '80301', '80401', '80402', '80403', '80499', '80501', '80502', '80601', '80701', '80702', '89999', '90101', '90102', '90103', '90104', '90199', '90201', '90202', '90299', '90301', '90302', '90399', '90401', '90501', '90502', '90599', '99999', '100101', '100102', '100103', '100199', '100201', '100299', '100304', '100305', '100401', '110101', '110201', '120101', '120201', '120202', '120299', '120301', '120302', '120303', '120304', '120305', '120306', '120307', '120308', '120309', '120310', '120311', '120312', '120313', '120399', '120401', '120402', '120403', '120404', '120405', '120499', '120501', '120502', '120503', '120504', '130101', '130102', '130103', '130104', '130105', '130106', '130107', '130108', '130109', '130110', '130112', '130113', '130114', '130115', '130116', '130117', '130118', '130119', '130120', '130122', '130124', '130125', '130126', '130127', '130128', '130129', '130130', '130131', '130132', '130133', '130134', '130135', '130136', '130199', '130202', '130203', '130207', '130210', '130212', '130213', '130216', '130218', '130219', '130220', '130222', '130223', '130224', '130225', '130226', '130227', '130229', '130232', '130299', '130301', '130302', '139999', '140101', '140102', '140103', '140105', '149999', '150101', '150102', '150103', '150104', '150105', '150106', '150199', '150201', '150202', '150203', '150204', '150299', '150301', '150302', '150399', '150401', '150402', '150501', '150601', '150602', '150699', '150701', '150801', '159999', '160101', '160102', '160103', '160104', '160105', '160106', '160107', '160108', '160199', '160201', '180101', '180201', '180202', '180203', '180204', '180205', '180206', '180207', '180208', '180209', '180299', '180301', '180302', '180303', '180304', '180305', '180401', '180402', '180403', '180404', '180405', '180499', '180501', '180502', '180503', '180504', '180599', '180601', '180602', '180603', '180604', '180699', '180701', '180702', '180703', '180704', '180801', '180802', '180803', '180804', '180805', '180806', '180807', '180899', '180901', '180902', '180903', '180904', '180905', '180999', '181001', '181002', '181101', '181201', '181202', '181203', '181204', '181205', '181299', '181301', '181302', '181399', '181401', '181501', '181599', '181601', '181801', '189999', '500101', '500103', '500105', '500106', '500107']