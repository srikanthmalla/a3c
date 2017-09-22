# A3C (Asynchronous Advantage Actor Critic):
Paper: https://arxiv.org/pdf/1602.01783v1.pdf

1. Run the main file:    `python main.py`

2. To customize, change the `params.py` file.

3. TF Records are created in folder `graphs/results`

4. If you want to record video automatically, change param `create_video=True` in `params.py`. This works when there is only one thread (use it when testing). 

5. #### Status: 
Image stacking needs to be done (otherwise it will be one to many mapping), use simplified network (currently using VGG).
