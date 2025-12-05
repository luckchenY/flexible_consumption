You can use ppo_moe_run.py to reproduce the results.
You should use "--env_name router-rule" first. Then you will get a model in /outputs. 
Then you can use these models to initialize the model in the next training (input the model path in ppo_moe_run.py).  
If you want to train the subpolicy and router together, use --env_name standard
