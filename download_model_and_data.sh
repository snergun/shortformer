wget https://dl.fbaipublicfiles.com/mega/wt103.zip
wget https://dl.fbaipublicfiles.com/shortformer/wikitext103-shortformer.pt
mkdir -p data
mkdir -p checkpoints/shortformer
unzip wt103_data_bin.zip -d data
rm wt103_data_bin.zip
mv wikitext103-shortformer.pt checkpoints/shortformer/model.pt