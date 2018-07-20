import configparser;


def defaultconfig():

	config = configparser.ConfigParser()

	config['test'] = { 'v1':1,'v2':'y','v3':100.1 }

	with open('config.ini','w') as conffile:
		config.write(conffile);


def parseconfig():

	pass;