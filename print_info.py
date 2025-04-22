
def main():
    from train import init
    func, config = init()
    train_config = config = config['inference']
    import torchvision.models
    import torch
    from torchstat import stat
    model = train_config['net']
    device = torch.device('cpu')
    model.to(device)
    stat(model.to(device), (3, 224, 224))


if __name__ == '__main__':
    main()