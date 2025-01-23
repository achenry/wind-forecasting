import lightning as L
trainer = L.Trainer(accelerator="gpu", devices=2, num_nodes=1)
