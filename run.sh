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
				  --gen_size 200;
		done
	done
done
