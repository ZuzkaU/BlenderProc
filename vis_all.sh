for fn in `find $1 -type f -name *.hdf5`; do
	blenderproc vis hdf5 $fn --save `dirname $fn`;
done
