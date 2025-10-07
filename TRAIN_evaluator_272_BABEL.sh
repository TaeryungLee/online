export HF_ENDPOINT=https://hf-mirror.com
cd Evaluator_272
# huggingface-cli download --resume-download distilbert/distilbert-base-uncased --local-dir ./deps/distilbert-base-uncased
ln -s ../data/babel_272 ./datasets/babel_272
python -m train --cfg configs/configs_evaluator_272/H3D-TMR_BABEL.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug
cd ..