redwood/tasks:
	for i in `ls ../../datasets/redwood/*.obj | sed 's/_.*//g' | uniq`; do \
		sid=`echo $$i | sed 's/.*\///g'`; \
        for j in `seq 0 49`; do \
			inc=$$(( j + 1)); \
			for k in `seq $${inc} 49`; do \
				name=$${sid}_$${j}_$${k}; \
				echo ./Super4PCS -i $${i}_$${j}.obj $${i}_$${k}.obj -o 0.7 -d 0.01 -t 1000 -n 200 -m redwood/super4pcs/$${name}.txt >> $@; \
			done \
        done \
	done

redwood/tasks.0.sh: redwood/tasks
	python split.py redwood/tasks 100

redwood: redwood/tasks redwood/tasks.0.sh
	make -p redwood/logs
	#condor_submit condor.sh  

scannet/tasks:
	python generate_tasks.py --dataset scannet

scannet/split:
	python split.py scannet/tasks 120
	chmod +x scannet/*.sh

scannet: scannet/tasks scannet/tasks.0.sh
	make -p scannet/logs
	#condor_submit condor.sh  
