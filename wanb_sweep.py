import wandb


def train(config):
    pass


def main():
    wandb.init(project="my-first-sweep")
    score = train(wandb.config)
    wandb.log({"score": score})


sweep_config = {
    'method': 'grid',
    "metric": {"goal": "minimize", "name": "val_loss"},
    'parameters': {
        'seed': {
            'values': [42, 43, 44, 45, 46]
        },
        'batch_size': {
            'values': [16, 32]
        },
        'learning_rate': {
            'min': 0.0000001,
            'max': 0.0001
        },

        'weight_decay': {
            'min': 0.0000001,
            'max': 0.0001
        },
        'StepLR_step_size': {
            'values': [10, 20, 30]
        },
        'StepLR_gamma': {
            'values': [0.1, 0.5, 0.9]
        },
        'augment_mode': {
            'values': ['rawboost', 'audiomentations']
        },

    }
}

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=main)
