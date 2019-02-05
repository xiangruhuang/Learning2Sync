all:
	make redwood.fgr.log
	make redwood.super4pcs.log
	make scannet.fgr.log
	make scannet.super4pcs.log

clean:
	rm -f redwood.fgr.log
	rm -f redwood.super4pcs.log
	rm -f scannet.fgr.log
	rm -f scannet.super4pcs.log	

redwood.fgr.log:
	python measure.py --dataset redwood --source fgr > $@

redwood.super4pcs.log:
	python measure.py --dataset redwood --source super4pcs > $@

scannet.fgr.log:
	python measure.py --dataset scannet --source fgr > $@

scannet.super4pcs.log:
	python measure.py --dataset scannet --source super4pcs > $@
