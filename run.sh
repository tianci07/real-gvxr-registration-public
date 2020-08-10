echo "metrics,target_image,algorithm,downscale,pop_size,nb_generations,runtime_sec,parameters,ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD" > results.csv

for m in {ZNCC,SSIM,MI,GC,MAE,CS,SSD,GD};
do
	for i in {1..3};
	do
		for d in {1..4};
		do ./optimisation.py --target_image ./00382-s1-neg2.png\
				  --output ./results\
				  --algorithm CMAES\
				  --repeat "$i"\
				  --metrics "$m"\
				  --downscale "$d"\
				  --pop_size 200\
				  --gen_size 200 > results-metrics$m-downscale$d-run$i.out 2> results-metrics$m-downscale$d-run$i.err
				  
		cat results-metrics$m-downscale$d-run$i.out >> results.csv
		done
	done
done
