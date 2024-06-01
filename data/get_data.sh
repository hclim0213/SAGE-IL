#dl_zinc:
curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/all.txt --create-dirs -o ./datasets/zinc/all.txt
curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/train.txt --create-dirs -o ./datasets/zinc/train.txt
curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/valid.txt --create-dirs -o ./datasets/zinc/valid.txt
curl https://raw.githubusercontent.com/wengong-jin/icml18-jtnn/master/data/zinc/test.txt --create-dirs -o ./datasets/zinc/test.txt

#dl_guacamol:
curl https://ndownloader.figshare.com/files/13612745 -L --create-dirs -o ./datasets/guacamol/all.txt
curl https://ndownloader.figshare.com/files/13612760 -L --create-dirs -o ./datasets/guacamol/train.txt
curl https://ndownloader.figshare.com/files/13612766 -L --create-dirs -o ./datasets/guacamol/valid.txt
curl https://ndownloader.figshare.com/files/13612757 -L --create-dirs -o ./datasets/guacamol/test.txt

#dl_zinc_pretrain:
curl https://raw.githubusercontent.com/sungsoo-ahn/genetic-expert-guided-learning/main/resource/checkpoint/zinc/generator_config.json --create-dirs -o ./pretrained_models/original_benchmarks/zinc/generator_config.json
curl https://github.com/sungsoo-ahn/genetic-expert-guided-learning/blob/main/resource/checkpoint/zinc/generator_weight.pt?raw=true -L --create-dirs -o ./pretrained_models/original_benchmarks/zinc/generator_weight.pt

#dl_guacamol_pretrain:
curl https://raw.githubusercontent.com/sungsoo-ahn/genetic-expert-guided-learning/main/resource/checkpoint/guacamol/generator_config.json --create-dirs -o ./pretrained_models/original_benchmarks/guacamol/generator_config.json
curl https://github.com/sungsoo-ahn/genetic-expert-guided-learning/blob/main/resource/checkpoint/guacamol/generator_weight.pt?raw=true -L --create-dirs -o ./pretrained_models/original_benchmarks/guacamol/generator_weight.pt
