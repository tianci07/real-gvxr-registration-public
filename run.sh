echo "flag,metrics,target_image,algorithm,downscale,img_width,img_height,pop_size,nb_generations,runtime_sec,ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD" > ./results/results.csv

MPLBACKEND="Agg";
touch ./results/initial-guess.txt;

for m in {ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD};
do
	for i in {1..3};
	do
		for d in {4..1};
		do ./optimisation.py --target_image 00382-s1-neg2.png\
				  --output ./results\
				  --algorithm CMAES\
				  --repeat "$i"\
				  --metrics "$m"\
				  --downscale "$d"\
				  --pop_size 1000\
				  --gen_size 1000\
					--initial_guess ./results/initial-guess.txt > ./results/results-metrics$m-downscale$d-run$i.out 2> ./results/results-metrics$m-downscale$d-run$i.err

		grep -w "METRICS" ./results/results-metrics$m-downscale$d-run$i.out >> ./results/results.csv
		grep -w "PARAMS" ./results/results-metrics$m-downscale$d-run$i.out > ./results/initial-guess.txt
		grep "FITNESS" ./results/results-metrics$m-downscale$d-run$i.out > ./results/fitness.txt

		./plot.py --repeat "$i"\
						--metrics "$m"\
						--downscale "$d"

		done
	done
done
