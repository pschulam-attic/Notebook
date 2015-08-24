SCRIPT=score_posteriors.R
TIMES='1.0 2.0 4.0'

for T in $TIMES
do

    echo "Predicting at ${T} years"
    Rscript $SCRIPT $T "posteriors_${T}.csv"
done
