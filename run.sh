# CMAES

if [ ! -d results ];
then
    mkdir results
fi

echo "flag,metrics,target_image,algorithm,downscale,img_width,img_height,pop_size,nb_generations,runtime_sec,ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD,ZNCC_f,SSIM_f,MI_f,GC_f,MAE_f,CS_f,SSD_f,GD_f" > ./results/results.csv

# MPLBACKEND="Agg";

# CMAES
# mkdir -p results/PARAMS/ results/FITNESS/ results/GENERATIONS/

#touch results/initial-guess.txt;

# NSGA-II
# mkdir -p results/MULTIPARAMS/ results/FITNESS/

#ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD
# for target in {00135-s1-3,00648-s1-3,00872-s1-1,00997-s1-1,01019-s1-1,01067-s1-1,01124-s1-2,01159-s1-2,01294-s2-2,01474-s1-2,01475-s1-1,01726-s1-2,01853-s2-1,01895-s1-2,02132-s1-1}

# # Image 2, with lowest MAE
# for target in 00648-s1-3;

# Image 10, with highest MAE
for target in 01474-s1-2;

# # Image 3, median results
# for target in 00872-s1-1;
do
    for m in MAE;
    do
	    for i in {1..3};
	    do
		    for d in {0..0};
		    do
			    echo $m $i $d
			    DIR=results/$target

			    mkdir -p $DIR $DIR/PARAMS/ $DIR/FITNESS/ $DIR/GENERATIONS/

			    touch $DIR/initial-guess.txt

			    lscpu |grep "Model name:"> $DIR/runtime
			    glxinfo |grep "OpenGL renderer string:" >> $DIR/runtime
			    grep MemTotal /proc/meminfo  >> $DIR/runtime

			    date >> $DIR/runtime-$i

					./optimisation.py --target_image $target.png\
                    --output $DIR/\
                    --algorithm CMAES\
                    --repeat "$i"\
                    --metrics "$m"\
                    --downscale "$d"\
                    --pop_size 50\
                    --gen_size "NONE"\
                    --initial_guess $DIR/initial-guess.txt > $DIR/output 2> $DIR/error
			    date >> $DIR/runtime-$i

		        # CMAES
                grep -w "METRICS" $DIR/output >> results/results.csv
                grep -w "METRICS" $DIR/output >> $DIR/results.txt
                # grep -w "INITIALGUESS" $DIR/output > results/initial-guess.txt
                grep -w "PARAMS" $DIR/output >> $DIR/PARAMS/parameters-$m-$i.txt
                grep -w "FITNESS" $DIR/output >> $DIR/FITNESS/fitness-$m-$i.txt
                # grep -w "FULLFITNESS" $DIR/output >> results/FULLFITNESS/fullfitness-$m-$i.txt
                grep -w "GENERATIONS" $DIR/output >> $DIR/GENERATIONS/nbgen-$m-$i.txt
                # rm $DIR/output $DIR/error

                # NSGA-II
                # grep -w "MULTIPARAMS" $DIR/output >> results/MULTIPARAMS/multiparameters-$m-$i.txt
                # grep -w "FITNESS" $DIR/output >> results/FITNESS/fitness-$m-$i.txt

		    done
	    # ./plot.py --repeat 1\
	    #   	--metrics MAE\
	    #   	--plot single_line
	    done
    done
done
