#while [ $(basename &(pwd)) -neq td-stargan-vc ]; do cd ..; done

echo $0 $1 $2 $3

model_dir=$1
out_dir=$2

epoch=200
device=$3

scrip_dir=test_scripts/vctk
mosnet_dir=../MOSNet

#CUDA_VISIBLE_DEVICES=$device python generate_from_dataset.py --save_path $out_dir/signals/ --load_path $model_dir --data_path datasets/vctk/ --data_file test_files_mcd --epoch $epoch

cp $model_dir/config.yaml $out_dir
cp $model_dir/githash $out_dir
echo $epoch > $out_dir/epoch

#python $scrip_dir/test_mcd.py --test_path $out_dir/signals/ --save_file $out_dir/mcd_results

CUDA_VISIBLE_DEVICES=$device python $scrip_dir/test_speaker_rec.py --test_path $out_dir/signals/ --save_file $out_dir/spkrec_results --speechbrain_hparam $scrip_dir/speechbrain_model/sb_classifier_hparams.yaml

#CUDA_VISIBLE_DEVICES=$device python $mosnet_dir/custom_test.py --pretrained_model $mosnet_dir/output/mosnet.h5 --rootdir $out_dir/signals/
#mv $out_dir/signals/MOSnet_result_raw.txt $out_dir/mosnet_result_raw.txt
#python $scrip_dir/test_mosnet.py --test_path $out_dir/mosnet_result_raw.txt --save_file $out_dir/mosnet_results

#python test_scripts/get_model_info.py --model_dir $model_dir --save_file $out_dir/info

python $scrip_dir/test_gen_html.py --test_dir $out_dir --save_file $out_dir/index.html


