#data_path=
#./data/redwood/chair/shuffle
mkdir -p matching
mkdir -p logs
split=$1
nworker=$2
njobs=`cat joblist | wc -l`
njobs=$(( njobs - 1 ))
count=0
for n in `seq 0 ${njobs}`; do
	count=$(( count - 1 ))
	if [[ -e "matching/${i}.npy" ]]; then
		echo file exists
		continue
	fi
	if [[ `expr ${n} % ${nworker}` == "${split}" ]]; then
		echo shapeid=${i}, s=${s}, t=${t}, id=${id}
		#python -c "import cv2; print(cv2.__version__)"
		#python openRGBDCondor.py --shapeid ${i} --split ${id}
	else
		echo skipping since parallelization
		continue
	fi
done
