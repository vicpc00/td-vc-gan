# TD-VC-GAN - Pitch-controled end-to-end voice conversion
[Published Paper](https://jcis.sbrt.org.br/jcis/article/view/896)
[Example page](https://vicpc00.github.io/td-vc-gan/)
## Prerequisites
1. Presequisites can be found on requirements.txt
2. Download WavLM Large model from [here](https://github.com/microsoft/unilm/tree/master/wavlm) to ./wavlm folder

## Prepare dataset
If using CETUC dataset, follow instructtions [here](https://github.com/vicpc00/filtered_cetuc_dataset)

```
python scripts/prepare_dataset.py [dataset files folder] --save_folder [dataset folder] --test_size 30 --ext '.wav' 
```

## Training
```
#Stage 1 - Reconstructtion
python train.py --save_path [stage 1 checkpoint folder] --data_path [dataset folder] --config_file config/wavml-stage1.yaml
#Stage 2-1 - Conversion
python train.py --save_path [stage 2-1 checkpoint folder] --data_path [dataset folder] --config_file config/wavml-stage2_1.yaml --load_path [stage 1 checkpoint folder]
#Stage 2-2 - Conversion with reverse
python train.py --save_path [stage 2-2 checkpoint folder] --data_path [dataset folder] --config_file config/wavml-stage2_2.yaml --load_path [stage 2-1 checkpoint folder]
```

# Generating signals
```
python generate_with_target.py --save_path [save path] --load_path [checkpoint path] --data_path [dataset path] --data_file test_files --epoch [epoch] --data_format alcaim
```
