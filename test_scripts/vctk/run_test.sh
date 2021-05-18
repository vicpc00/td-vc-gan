#while [ $(basename &(pwd)) -neq td-stargan-vc ]; do cd ..; done

model_dir=$1
out_dir=$2

epoch=200

scrip_dir=test_scripts/vctk

CUDA_VISIBLE_DEVICES=1 python generate_from_dataset.py --save_path $out_dir/signals/ --load_path $model_dir --data_path datasets/vctk/ --data_file test_files_mcd --epoch $epoch

cp $model_dir/config.yaml $out_dir
cp $model_dir/githash $out_dir
echo $epoch > $out_dir/epoch

python $scrip_dir/test_mcd.py --test_path $out_dir/signals/ --save_file $out_dir/dists_mcd

python $scrip_dir/test_speaker_rec.py --test_path $out_dir/signals/ --save_file $out_dir/spkrec_results --speechbrain_hparam $scrip_dir/speechbrain_model/sb_classifier_hparams.yaml

python test_scripts/get_model_info.py --model_dir $model_dir --save_file $out_dir/info

python $scrip_dir/test_gen_html.py --test_dir $out_dir --save_file $out_dir/index.html


