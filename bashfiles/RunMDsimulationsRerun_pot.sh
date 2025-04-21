

export GMX_MAXBACKUP=-1

#CREATE outputs for all the molecules using a forloop

#NEED: directory with all pdb files and directory with all mol2 files
#how to get these? via an website and some code

#for i in {001..010} #to run molecule 1 to 10
#todo= {006,042,076}
# todo2= {006,078,095,125,128,129,150,171,176,183,186,218,223,229,233,237,238,244,248,255,260,262,263,264,265,294,297,321,334,337,341,361,383,385,391,392,407,411,412,444,460,462,466,467,474,485,492,496,508,511,517,523,530,533,538,548,556,565,569,587,588,600,602,603,615}

#echo "Fixed using array:"
#a=(001 002 003)
#for x in "${a[@]}"

echo go
#./RunMDsimulations.sh >> testing.log 2>&1
# for i in {003,006,015,016,043,052,054,055,056,062,070,080,086,094,109,115,199,203,220,224,225,229,230,246,248,249,250,255,259,275,279,291,292,307,309,320,338,353,365,370,374,376,377,392,395,420,426,428,436,443,449,453,478,480,498,511,514,517,519,520,523,538,539,542,551,563,566,579,580,581,586,588,601,602,609,621,623,624,628,629,636,637,641,644,649,658,668,669,680,689,702,705,706,707,711,721,722,724,728,729,733,734,739,746,750,754,759,769,781,782,783,784,788,794,811,815,834,845,852,855}
DATASET=GSK3
for i in {002,764,765}
do
    echo %%%%%%%%%%%%%%%%%         NEXT MOLECULE         %%%%%%%%%%%%%%%%%%%%%%%%%%%
    echo MDsimulations_${DATASET}/$i
    if [ -d "MDsimulations_${DATASET}/$i" ]; then
        cd "MDsimulations_${DATASET}/$i"

        #production
        LIGPDB=${i}_gmx.pdb
        LIGTOP=${i}_gmx.top
        TPRFILE=${i}_prod_ligandonly.tpr
        OUTPUTFILE=${i}_prod_ligonly

        #make index file!
        echo q | gmx make_ndx -f ${i}_prod.tpr -o ${i}_index.ndx

        echo 2 | gmx trjconv -s ${i}_prod.tpr -f ${i}_prod.xtc -n ${i}_index.ndx -o ${i}_prod_ligonly.xtc
        echo 2 | gmx trjconv -s ${i}_prod.tpr -f ${i}_prod.xtc -n ${i}_index.ndx -o ${i}_prod_ligonly.pdb -dump 0

        #copy paste 001_gmx to here from setup
        cp ./setup${i}/${i}_gmx.top .

        gmx grompp -f ${SILCSBIODIR}/data/gromacs/gmx_inputs/prod_rerun_pot.mdp -c ${i}_prod_ligonly.pdb -p ${LIGTOP} -o ${TPRFILE}
        gmx mdrun -s ${TPRFILE} -rerun ${i}_prod_ligonly.xtc -e ${OUTPUTFILE}
        echo Potential | gmx energy -f ${i}_prod_ligonly.edr -o ${i}_lig_pot_energy.xvg
        cd ../..
    else
        echo "Directory MDsimulations_${DATASET}/$i does not exist."
    fi
done
