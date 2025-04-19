# stage 1: contrastive learning
python main_AsyCon.py --config configs/mini.yml epoch 500 lamb 0.5

# stage 2: finetune 
python main_ft.py --config configs/mini.yml model_path xxx epoch 50 w_dis 2.0
```