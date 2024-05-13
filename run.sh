yaml_list=(configs/config_skin_isic2016.yml configs/config_skin_isic2017.yml configs/config_skin_isic2018.yml);

for yaml in "${yaml_list[@]}"; do
    python train.py --config "$yaml";
    python test.py --config "$yaml";
done

zip -r result.zip result && mv result.zip /autodl-fs/data/LGANet/;
shutdown -h now;