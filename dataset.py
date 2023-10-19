import torch
from torchvision import transforms


def earpiece_dataloader(img, model_jit):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),

            transforms.ToTensor(),
            transforms.Normalize([0.4332, 0.2901, 0.2727], [0.2541, 0.2048, 0.1843]),
        ])


    img = transform(img)
    img = img.unsqueeze(0)

    use_cuda = torch.cuda.is_available()
    if torch.cuda.is_available():
        img = img.cuda()

    outputs = model_jit(img)

    _, preds = torch.max(outputs.data, 1)
    preds = preds.cpu().numpy() if use_cuda else preds.numpy()

    return preds

