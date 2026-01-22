ln -s ../utils ./Evaluator_272/
ln -s ../humanml3d_272 ./Evaluator_272/
ln -s ../options ./Evaluator_272/
ln -s ../models ./Evaluator_272/
ln -s ../visualization ./Evaluator_272/
ln -s ../Causal_TAE ./Evaluator_272/
python eval_t2m.py --resume-pth Causal_TAE/net_last.pth