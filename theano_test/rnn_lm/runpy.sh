nohup python run_lm.py --learning-rate=0.001 --vocab-size=5000 --hidden-dim=32 --embedding-dim=10 --print-every=5 --model=first.model --dataset=data/select_top_3000_cn > log 2>&1&
echo $! > run_lm.pid
echo "run_lm... pid:" `cat run_lm.pid`
