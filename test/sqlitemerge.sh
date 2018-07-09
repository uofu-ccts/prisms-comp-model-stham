#!/bin/bash
date
NEWDB=${1/.sqlite3/}.merge.sqlite3
echo $NEWDB

sqlite3 $1 '.schema acttraj' | sqlite3 $NEWDB

for f in "$@"
do
	date
	ls -lh $f
	sqlite3 $NEWDB "attach '$f' as tmerge; begin; insert into acttraj (\`index\`, length, start, long, lat, agentnum, actcode, day, day365) select \`index\`,length,start,long,lat,agentnum,actcode,day,day365 from tmerge.acttraj; commit; detach tmerge;"
done
date




