python3 main_distill.py --gpu_id 1 --seed 0 -d cora -t sage -s rbm -m transductive --config parameters.yaml --batch_size 4096 --reliable_sampling --adv_augment
python3 main_distill.py --gpu_id 1 --seed 0 -d pubmed -t sage -s rbm -m transductive --config parameters.yaml --batch_size 4096 --reliable_sampling
python3 main_distill.py --gpu_id 1 --seed 0 -d pubmed -t sage -s mlp -m inductive --config parameters.yaml --reliable_sampling
python3 main_distill.py --gpu_id 1 --seed 0 -d amazon-com -t sage -s mlp -m transductive --config parameters.yaml --positional_encoding --adv_augment
python3 main_distill.py --gpu_id 1 --seed 0 -d amazon-com -t sage -s mlp -m inductive --config parameters.yaml --positional_encoding --adv_augment
