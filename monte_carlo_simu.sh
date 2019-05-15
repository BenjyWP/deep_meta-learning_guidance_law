i=1
while(($i<=100))
do
    echo $i
    python mppi_run.py --iter_num=$i
    let "i++"
done