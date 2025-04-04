import lightning.pytorch as L

if __name__ == "__main__":
    trainer = L.Trainer(accelerator="gpu", devices=1, num_nodes=1)