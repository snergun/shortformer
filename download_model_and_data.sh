# Download Data
wget https://dl.fbaipublicfiles.com/mega/data/wt103_data_bin.zip
mkdir -p data
unzip wt103_data_bin.zip -d data
rm wt103_data_bin.zip
# Download Model Checkpoint
wget https://dl.fbaipublicfiles.com/shortformer/wikitext103-shortformer.pt
mv wikitext103-shortformer.pt checkpoints/shortformer/model.pt
mkdir -p checkpoints/shortformer

