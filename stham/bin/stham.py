#!/usr/bin/env python
# import configparser;
import argparse;
import stham.config

sthamversion = 'beta 0.1'

def main():
	#parse arguments
	parser = argparse.ArgumentParser(description="Runs the STHAM agent based exposure model.")

	parser.add_argument('-g','--gen',action='store_true',help="generates a default config")
	parser.add_argument('-c','--conf',type=str, help="specifies the config file; default is \'conf.ini\
	'" , default="conf.ini");
	parser.add_argument('--version',action='version',version=sthamversion,help='prints the version')

	args = parser.parse_args()

	#create default ini
	if(args.gen):
		#FIXME generate conf
		print("default configuration generated")
		print("remember to modify path variables!")
		return;

	#run ini
	#FIXME get dictionary
	stham.config.testy()
	#FIXME run model
	




if __name__=="__main__":
	main()


